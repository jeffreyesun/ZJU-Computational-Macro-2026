"""
Pure consumption-savings choice on the wealth grid. `b_in` and
`b_end` both live on the same continuous wealth axis; implied
consumption `c = b_in - b_end`. `utility(cell, c; env) -> T`; `-Inf`
or non-positive consumption are skipped. `monotone_search` ∈
`(:sequential, :divide_conquer)` selects the inner-loop strategy
(D&C requires non-decreasing optimal policy in input wealth — true
under concave-u + linear-budget).
"""
struct ConsumptionSavingsStageSpec{F_u, T, Search} <: AbstractStageSpec
    β           :: T
    utility     :: F_u
    wealth_axis :: Symbol
end

function ConsumptionSavingsStageSpec(; β, utility, wealth_axis::Symbol=:wealth,
                                     monotone_search::Symbol=:sequential)
    @assert monotone_search in (:sequential, :divide_conquer)
    return ConsumptionSavingsStageSpec{typeof(utility), typeof(β), Val{monotone_search}}(
        β, utility, wealth_axis,
    )
end

"Kernel: chosen `b_end` index per cell."
struct ConsumptionSavingsKernel{P<:AbstractArray{Int}}
    policy :: P
end

allocate_kernel(::ConsumptionSavingsStageSpec, ::Type, layout::StateLayout) =
    ConsumptionSavingsKernel(zeros(Int, layout_size(layout)))

# Backward (monotone-policy argmax) #
#-----------------------------------#

# Sequential walk: search for each cell starts at previous cell's argmax.
function _cs_backward_slice!(::Type{Val{:sequential}},
                             spec::ConsumptionSavingsStageSpec{F_u, T},
                             V_end, env, V_start, policy,
                             layout::StateLayout, wdim, wgrid, n_w, β,
                             names, axvals,
                             other_idxs::NTuple{N, Int}) where {F_u, T, N}
    prev_a = 1
    for w_in_i in 1:n_w
        in_idxs = ntuple(i -> i == wdim ? w_in_i : other_idxs[i], N)
        ci_in   = CartesianIndex(in_idxs)
        cell    = NamedTuple{names}(ntuple(i -> axvals[i][in_idxs[i]], N))
        b_in    = wgrid[w_in_i]

        best_v = typemin(T)
        best_a = 0
        for a_i in prev_a:n_w
            c = b_in - wgrid[a_i]
            c > 0 || continue
            u = spec.utility(cell, c; env=env)
            isfinite(u) || continue
            ci_out = set_coord(ci_in, layout, spec.wealth_axis => a_i)
            v = u + β * V_end[ci_out]
            if v > best_v
                best_v = v
                best_a = a_i
            end
        end

        if best_a == 0
            V_start[ci_in] = typemin(T)
            policy[ci_in]  = 1
        else
            V_start[ci_in] = best_v
            policy[ci_in]  = best_a
            prev_a         = best_a
        end
    end
    return
end

# Divide-and-conquer: O(n_w log n_w) under MPS assumption.
function _cs_backward_slice!(::Type{Val{:divide_conquer}},
                             spec::ConsumptionSavingsStageSpec{F_u, T},
                             V_end, env, V_start, policy,
                             layout::StateLayout, wdim, wgrid, n_w, β,
                             names, axvals,
                             other_idxs::NTuple{N, Int}) where {F_u, T, N}
    function _rec!(lo::Int, hi::Int, lo_b::Int, hi_b::Int)
        lo > hi && return
        mid = (lo + hi) >> 1
        in_idxs = ntuple(i -> i == wdim ? mid : other_idxs[i], N)
        ci_in   = CartesianIndex(in_idxs)
        cell    = NamedTuple{names}(ntuple(i -> axvals[i][in_idxs[i]], N))
        b_in    = wgrid[mid]

        best_v = typemin(T)
        best_a = 0
        for a_i in lo_b:hi_b
            c = b_in - wgrid[a_i]
            c > 0 || continue
            u = spec.utility(cell, c; env=env)
            isfinite(u) || continue
            ci_out = set_coord(ci_in, layout, spec.wealth_axis => a_i)
            v = u + β * V_end[ci_out]
            if v > best_v
                best_v = v
                best_a = a_i
            end
        end

        if best_a == 0
            V_start[ci_in] = typemin(T)
            policy[ci_in]  = 1
            _rec!(lo, mid - 1, lo_b, hi_b)
            _rec!(mid + 1, hi, lo_b, hi_b)
        else
            V_start[ci_in] = best_v
            policy[ci_in]  = best_a
            _rec!(lo, mid - 1, lo_b, best_a)
            _rec!(mid + 1, hi, best_a, hi_b)
        end
        return
    end
    _rec!(1, n_w, 1, n_w)
    return
end

function backward!(buffer, spec::ConsumptionSavingsStageSpec{F_u, T, Search},
                   V_end, env) where {F_u, T, Search}
    (;input_layout, dims, V_start, β) = resolve(buffer, spec, env)
    wealth_dim = axis_dim(input_layout, spec.wealth_axis)
    policy     = buffer.kernel.policy
    wgrid      = axisvalues(input_layout.axes[wealth_dim])
    n_w        = length(wgrid)

    N           = ndims(V_start)
    names       = axisnames(input_layout)
    axvals      = ntuple(i -> axisvalues(input_layout.axes[i]), N)
    other_sizes = ntuple(i -> i == wealth_dim ? 1 : dims[i], N)

    for other_ci in CartesianIndices(other_sizes)
        _cs_backward_slice!(Search, spec, V_end, env, V_start, policy,
                            input_layout, wealth_dim, wgrid, n_w, β,
                            names, axvals, other_ci.I)
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward #
#---------#

function forward!(buffer, spec::ConsumptionSavingsStageSpec, Λ_start)
    (;input_layout, Λ_end) = resolve(buffer, spec)
    policy = buffer.kernel.policy
    T      = eltype(Λ_end)

    fill!(Λ_end, zero(T))
    for (idx, _) in cells(input_layout)
        ci_in = CartesianIndex(Tuple(idx))
        mass  = Λ_start[ci_in]
        iszero(mass) && continue
        ci_out = set_coord(ci_in, input_layout, spec.wealth_axis => policy[ci_in])
        Λ_end[ci_out] += mass
    end
    return Λ_end
end

# Wrapper #
#---------#

@definestage ConsumptionSavingsStage ConsumptionSavingsStageSpec kernel=ConsumptionSavingsKernel
