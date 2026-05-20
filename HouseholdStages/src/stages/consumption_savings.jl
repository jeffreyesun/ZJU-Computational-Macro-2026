"""
Configuration for the pure consumption-savings choice on the wealth
grid. The state space is unchanged across the stage: input wealth
`b_in` and output wealth `b_end` both live on the same continuous
wealth axis (a `ContinuousGrid` axis whose values are the discrete
grid). The household picks `b_end` on that grid; the implied
consumption is `c = b_in - b_end`.

User closure (cell-positional, env-kwarg): `utility(cell, c; env) -> T`.
`-Inf` and non-positive consumption are skipped. `β::Param{T}` is the
discount factor (calibrated or env-swept).

# Inner-loop search strategy

`monotone_search` selects how the per-slice argmax is computed and
is carried as a `Val`-typed type parameter on the Spec for type-stable
dispatch:

  * `:sequential` (default) — the cell-by-cell walk along the wealth
    axis with `prev_a` as the lower bound. Total work per slice
    `O(n_w · n_w)` worst-case, `O(n_w)` when the policy advances
    monotonically. No correctness assumption beyond what the user's
    `utility` enforces.
  * `:divide_conquer` — divide-and-conquer monotone-policy argmax.
    Total work per slice `O(n_w log n_w)`, strictly better than
    sequential at large `n_w`. **Requires that the optimal policy be
    non-decreasing in input wealth — i.e., always-non-negative marginal
    propensity to save.** Concave utility + linear budget implies this,
    but non-convex feasibility or hand-built non-concave payoffs can
    violate it.

The two paths produce byte-identical policies under the MPS assumption;
under a violation, `:divide_conquer` may converge to a different local
optimum on a non-monotone cell.

This stage is the "consumption-savings" piece of the L03 / L04
decomposition: pair it with a [`WealthChangeStage`](@ref) for income
receipt to recover the Aiyagari period structure
`IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavingsStage`.

Pure data — no per-call buffers.
"""
struct ConsumptionSavingsStageSpec{F_u, T<:Real,
                                   LIn<:StateLayout, LOut<:StateLayout,
                                   Search} <: AbstractStageSpec
    β               :: Param{T}
    utility         :: F_u
    wealth_axis     :: Symbol
    wealth_dim      :: Int
    input_layout    :: LIn
    output_layout   :: LOut
    element_type    :: Type{T}
    monotone_search :: Search
end

"""
    ConsumptionSavingsStageSpec(layout; β, utility, wealth_axis=:wealth,
                                monotone_search=:sequential,
                                element_type=nothing)

Build the Spec for a [`ConsumptionSavingsStage`](@ref). `utility` is
`(cell, c; env) -> T`; `β` is a [`Param`](@ref) or a raw number;
`monotone_search` is `:sequential` (default) or `:divide_conquer` —
see the struct docstring for the MPS caveat. `element_type` defaults
to the eltype derived from `β`'s value (or `Float64` for a
Symbol-valued sweep key).
"""
function ConsumptionSavingsStageSpec(layout::StateLayout;
                                     β,
                                     utility,
                                     wealth_axis::Symbol = :wealth,
                                     monotone_search::Symbol = :sequential,
                                     element_type::Union{Type, Nothing} = nothing)
    monotone_search in (:sequential, :divide_conquer) ||
        error("ConsumptionSavingsStageSpec: monotone_search must be " *
              ":sequential or :divide_conquer, got :$monotone_search")
    wealth_dim = axis_position(layout, wealth_axis)
    β_param = β isa Param ? β : Param(Float64(β))

    T_default = let v = β_param.val
        if v isa Symbol
            Float64
        else
            (typeof(v) <: Real) ? typeof(v) : Float64
        end
    end
    T = @something element_type T_default

    ms_val = Val(monotone_search)
    return ConsumptionSavingsStageSpec{typeof(utility), T,
                                       typeof(layout), typeof(layout),
                                       typeof(ms_val)}(
        β_param, utility, wealth_axis, wealth_dim,
        layout, layout, T, ms_val,
    )
end

"""
Per-call buffer for a consumption-savings stage. The kernel is a
NamedTuple `(; policy::Array{Int,N})` holding the chosen-`b_end` index
per cell. No scratch state is needed beyond V_start / Λ_end.
"""
struct ConsumptionSavingsStageBuffer{T<:Real, N, AV<:AbstractArray{T,N},
                                     Kernel} <: AbstractStageBuffer
    kernel  :: Kernel
    scratch :: Nothing
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A pure consumption-savings choice on the wealth grid. Construct via
`ConsumptionSavingsStage(layout; β, utility, wealth_axis=:wealth,
monotone_search=:sequential)`. Composes via `∘` and `×`. The
`monotone_search` policy is carried as a `Val`-typed type parameter
on the Spec.
"""
struct ConsumptionSavingsStage{Spec<:ConsumptionSavingsStageSpec,
                               Buffer<:ConsumptionSavingsStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function ConsumptionSavingsStage(layout::StateLayout;
                                 β,
                                 utility,
                                 wealth_axis::Symbol = :wealth,
                                 monotone_search::Symbol = :sequential,
                                 element_type::Union{Type, Nothing} = nothing,
                                 V_start::Union{Nothing, AbstractArray} = nothing,
                                 Λ_end::Union{Nothing, AbstractArray}  = nothing)
    spec = ConsumptionSavingsStageSpec(layout; β, utility, wealth_axis,
                                       monotone_search, element_type)
    return ConsumptionSavingsStage(spec, allocate(spec, spec.element_type;
                                                  V_start, Λ_end))
end

ConsumptionSavingsStage(spec::ConsumptionSavingsStageSpec) =
    ConsumptionSavingsStage(spec, allocate(spec))
bundle(spec::ConsumptionSavingsStageSpec) = ConsumptionSavingsStage(spec)

static_env_deps(::Type{<:ConsumptionSavingsStageSpec}) = NamedTuple()

# Allocate #
#----------#

function allocate(spec::ConsumptionSavingsStageSpec, ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    policy = zeros(Int, size(Vs))
    kernel = (; policy)
    return ConsumptionSavingsStageBuffer{T, ndims(Vs), typeof(Vs), typeof(kernel)}(
        kernel, nothing, Vs, Λe, CacheState(),
    )
end

# Backward (monotone-policy argmax) #
#-----------------------------------#
# Sequential walk: each cell's search starts at the previous cell's
# argmax and runs up to `n_w`. Total work bounded above by O(n_w · n_w);
# typically much less when policy advances rapidly.
function _cs_backward_slice!(::Type{Val{:sequential}},
                             spec::ConsumptionSavingsStageSpec{F_u,T},
                             V_end, env, V_start, policy,
                             wdim, wgrid, n_w, β,
                             names, axvals, dims,
                             other_idxs::NTuple{N,Int}) where {F_u,T,N}
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
            u = spec.utility(cell, c; env = env)
            isfinite(u) || continue
            out_idxs = Base.setindex(in_idxs, a_i, wdim)
            v = u + β * V_end[CartesianIndex(out_idxs)]
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

# Divide-and-conquer: recursively bisect the input-wealth axis, using
# each midpoint's argmax to bound its left and right sub-problems. Total
# work `O(n_w log n_w)` under the monotone-policy assumption.
function _cs_backward_slice!(::Type{Val{:divide_conquer}},
                             spec::ConsumptionSavingsStageSpec{F_u,T},
                             V_end, env, V_start, policy,
                             wdim, wgrid, n_w, β,
                             names, axvals, dims,
                             other_idxs::NTuple{N,Int}) where {F_u,T,N}
    # Recursive D&C with on-the-fly utility evaluation. `lo_b`, `hi_b`
    # bracket the possible argmax for any input-wealth index in `[lo..hi]`.
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
            u = spec.utility(cell, c; env = env)
            isfinite(u) || continue
            out_idxs = Base.setindex(in_idxs, a_i, wdim)
            v = u + β * V_end[CartesianIndex(out_idxs)]
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

function backward!(spec::ConsumptionSavingsStageSpec{F_u,T,LIn,LOut,Search},
                   V_end, env,
                   buffer::ConsumptionSavingsStageBuffer) where {F_u,T,LIn,LOut,Search}
    (; input_layout, wealth_dim) = spec
    V_start = buffer.V_start
    policy  = buffer.kernel.policy
    wgrid   = axisvalues(input_layout.axes[wealth_dim])
    n_w     = length(wgrid)
    β       = resolve(spec.β, env)

    N           = ndims(V_start)
    names       = axisnames(input_layout)
    axvals      = ntuple(i -> axisvalues(input_layout.axes[i]), N)
    dims        = layout_size(input_layout)
    other_sizes = ntuple(i -> i == wealth_dim ? 1 : dims[i], N)

    for other_ci in CartesianIndices(other_sizes)
        _cs_backward_slice!(Search, spec, V_end, env, V_start, policy,
                            wealth_dim, wgrid, n_w, β,
                            names, axvals, dims, other_ci.I)
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward #
#---------#

function forward!(spec::ConsumptionSavingsStageSpec, Λ_start,
                  buffer::ConsumptionSavingsStageBuffer)
    (; input_layout, wealth_dim) = spec
    Λ_end  = buffer.Λ_end
    policy = buffer.kernel.policy
    T      = eltype(Λ_end)

    fill!(Λ_end, zero(T))
    for (idx, _) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        mass    = Λ_start[ci_in]
        iszero(mass) && continue
        a_i = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a_i, wealth_dim)
        Λ_end[CartesianIndex(out_idxs)] += mass
    end
    return Λ_end
end
