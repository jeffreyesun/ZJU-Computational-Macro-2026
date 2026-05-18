"""
Pure consumption-savings choice on the wealth grid. The state space is
unchanged across the stage: input wealth `b_in` and output wealth
`b_end` both live on the same continuous wealth axis (a `ContinuousGrid`
axis whose values are the discrete grid). The household picks `b_end`
on that grid; the implied consumption is `c = b_in - b_end`.

User closure (cell-positional, env-kwarg): `utility(cell, c; env) -> T`.
`-Inf` and non-positive consumption are skipped. `β::Param{T}` is the
discount factor (calibrated or env-swept).

# Inner-loop search strategy

`monotone_search` selects how the per-slice argmax is computed:

  * `:sequential` (default) — the cell-by-cell walk along the wealth
    axis with `prev_a` as the lower bound. Total work per slice
    `O(n_w · n_w)` worst-case, `O(n_w)` when the policy advances
    monotonically. No correctness assumption beyond what the user's
    `utility` enforces.
  * `:divide_conquer` — divide-and-conquer monotone-policy argmax
    (port of `reference_materials/example_stages/helper/interpolations.jl`).
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
decomposition: pair it with a [`WealthChange`](@ref) for income receipt
to recover the Aiyagari period structure
`IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings`.
"""
struct ConsumptionSavings{F_u,
                          LIn<:StateLayout, LOut<:StateLayout,
                          T<:Real, N, D, AV<:AbstractArray{T,N},
                          Search} <: AbstractStage
    β               :: Param{T}
    utility         :: F_u
    wealth_axis     :: Symbol
    wealth_dim      :: Int
    closure_deps    :: NTuple{D, Symbol}
    input_layout    :: LIn
    output_layout   :: LOut
    V_start         :: AV
    Λ_end           :: AV
    policy          :: Array{Int, N}
    monotone_search :: Search
end

"""
    ConsumptionSavings(layout; β, utility, wealth_axis=:wealth,
                                  closure_deps=(),
                                  monotone_search=:sequential) -> ConsumptionSavings

Construct a [`ConsumptionSavings`](@ref) stage on `layout`. `utility` is
`(cell, c; env) -> T`; `β` is a [`Param`](@ref) or a raw number;
`monotone_search` is `:sequential` (default) or `:divide_conquer` — see
the struct docstring for the MPS caveat.
"""
function ConsumptionSavings(layout::StateLayout;
                            β,
                            utility,
                            wealth_axis::Symbol = :wealth,
                            closure_deps::NTuple{D, Symbol} = (),
                            monotone_search::Symbol = :sequential,
                            element_type::Union{Type, Nothing} = nothing,
                            Λ_end::Union{Nothing, AbstractArray}  = nothing) where {D}
    monotone_search in (:sequential, :divide_conquer) ||
        error("ConsumptionSavings: monotone_search must be :sequential or " *
              ":divide_conquer, got :$monotone_search")
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

    dims = layout_size(layout)
    Vs   = zeros(T, dims)
    Λe   = @something Λ_end zeros(T, dims)

    @assert typeof(Vs) === typeof(Λe) "ConsumptionSavings: V_start and Λ_end must have the same concrete array type"
    
    policy = zeros(Int, dims)
    ms_val = Val(monotone_search)

    return ConsumptionSavings{typeof(utility),
                              typeof(layout), typeof(layout),
                              T, length(dims), D, typeof(Vs),
                              typeof(ms_val)}(
        β_param, utility, wealth_axis, wealth_dim, closure_deps,
        layout, layout, Vs, Λe, policy, ms_val,
    )
end

static_env_deps(::Type{<:ConsumptionSavings}) = NamedTuple()

function allocate(stage::ConsumptionSavings{F_u,LIn,LOut,T},
                  ::Type{T2} = T) where {F_u,LIn,LOut,T,T2}
    return ((policy = stage.policy,), nothing)
end

# Backward (monotone-policy argmax) #
#-----------------------------------#
# Sequential walk: each cell's search starts at the previous cell's
# argmax and runs up to `n_w`. Total work bounded above by O(n_w · n_w);
# typically much less when policy advances rapidly.
function _cs_backward_slice!(::Type{Val{:sequential}},
                             stage::ConsumptionSavings{F_u,LIn,LOut,T,N},
                             V_end, env, V_start, policy,
                             layout, wdim, wgrid, n_w, β,
                             names, axvals, dims,
                             other_idxs::NTuple{N,Int}) where {F_u,LIn,LOut,T,N}
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
            u = stage.utility(cell, c; env = env)
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
                             stage::ConsumptionSavings{F_u,LIn,LOut,T,N},
                             V_end, env, V_start, policy,
                             layout, wdim, wgrid, n_w, β,
                             names, axvals, dims,
                             other_idxs::NTuple{N,Int}) where {F_u,LIn,LOut,T,N}
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
            u = stage.utility(cell, c; env = env)
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

function backward!(stage::ConsumptionSavings{F_u,LIn,LOut,T,N,D,AV,Search},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {F_u,LIn,LOut,T,N,D,AV,Search}
    layout   = stage.input_layout
    wdim     = stage.wealth_dim
    wgrid    = axisvalues(layout.axes[wdim])
    n_w      = length(wgrid)
    β        = resolve(stage.β, env)
    V_start  = stage.V_start
    policy   = stage.policy

    names   = axisnames(layout)
    axvals  = ntuple(i -> axisvalues(layout.axes[i]), N)
    dims    = layout_size(layout)
    other_sizes = ntuple(i -> i == wdim ? 1 : dims[i], N)

    for other_ci in CartesianIndices(other_sizes)
        _cs_backward_slice!(Search, stage, V_end, env, V_start, policy,
                            layout, wdim, wgrid, n_w, β,
                            names, axvals, dims, other_ci.I)
    end
    return V_start
end

# Forward #
#---------#
function forward!(stage::ConsumptionSavings{F_u,LIn,LOut,T,N},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {F_u,LIn,LOut,T,N}
    layout = stage.input_layout
    wdim   = stage.wealth_dim
    Λ_end  = stage.Λ_end
    policy = stage.policy

    fill!(Λ_end, zero(T))
    for (idx, _) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        mass    = Λ_start[ci_in]
        iszero(mass) && continue
        a_i = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a_i, wdim)
        Λ_end[CartesianIndex(out_idxs)] += mass
    end
    return Λ_end
end
