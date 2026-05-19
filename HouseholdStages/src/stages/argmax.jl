"""
Hard discrete choice over a categorical axis. Actions are the levels of
`choice_axis`. The K-operator is a sparse permutation (`δ_{π(s)}` for
each input cell, where `π(s)` is the chosen-action's next-cell index);
the kernel stores the integer policy.

`flow_payoff(cell, action; env)` is the period payoff at the cell of
taking action `action`; `-Inf` is treated as "unavailable" and skipped.
`next_state_idx(cell, action) -> Int` returns the integer index along
`choice_axis` of the cell reached by `action`.
"""
struct ArgmaxStage{F, BF,
              LIn<:StateLayout, LOut<:StateLayout,
              N, T<:Real, AV<:AbstractArray{T,N}} <: AbstractStage
    choice_axis    :: Symbol
    choice_dim     :: Int
    flow_payoff    :: F
    next_state_idx :: BF
    input_layout   :: LIn
    output_layout  :: LOut
    V_start        :: AV
    Λ_end          :: AV
    policy         :: Array{Int, N}
end

"""
    ArgmaxStage(layout; choice_axis, flow_payoff, next_state_idx,
                  element_type=Float64) -> ArgmaxStage

Construct an [`ArgmaxStage`](@ref) stage. `flow_payoff(cell, action; env)`
and `next_state_idx(cell, action)` are user closures (see the struct
docstring).
"""
function ArgmaxStage(layout::StateLayout;
                choice_axis::Symbol,
                flow_payoff,
                next_state_idx,
                element_type::Type{T} = Float64,
                V_start::Union{Nothing, AbstractArray} = nothing,
                Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    choice_dim = axis_position(layout, choice_axis)
    (; Vs, Λe) = _alloc_VΛ(layout, T, V_start, Λ_end)
    N          = ndims(Vs)
    policy     = zeros(Int, size(Vs))
    return ArgmaxStage{typeof(flow_payoff),
                  typeof(next_state_idx),
                  typeof(layout), typeof(layout),
                  N, T, typeof(Vs)}(
        choice_axis, choice_dim, flow_payoff, next_state_idx,
        layout, layout, Vs, Λe, policy,
    )
end

static_env_deps(::Type{<:ArgmaxStage}) = NamedTuple()

allocate(stage::ArgmaxStage, ::Type = Float64) =
    (; kernel = (; policy = stage.policy), scratch = nothing)

# Backward #
#----------#

function backward!(stage::ArgmaxStage, V_end, env, buffers)
    (; kernel, scratch) = buffers
    (; input_layout, choice_dim, V_start, policy) = stage
    actions = axisvalues(input_layout.axes[choice_dim])
    T = eltype(V_start)

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)

        best_v   = typemin(T)
        best_a_i = 0
        for (a_i, action) in pairs(actions)
            r = stage.flow_payoff(cell, action; env = env)
            isfinite(r) || continue
            next_axis_i = stage.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            v = r + V_end[CartesianIndex(out_idxs)]
            if v > best_v
                best_v   = v
                best_a_i = a_i
            end
        end
        best_a_i == 0 && error("ArgmaxStage: no finite-payoff action at cell $cell")

        V_start[ci_in] = best_v
        policy[ci_in]  = best_a_i
    end
    return V_start
end

# Forward #
#---------#

function forward!(stage::ArgmaxStage, Λ_start, buffers, moments = nothing)
    (; kernel, scratch) = buffers
    (; input_layout, choice_dim, Λ_end, policy) = stage
    actions = axisvalues(input_layout.axes[choice_dim])
    T = eltype(Λ_end)

    fill!(Λ_end, zero(T))
    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        mass    = Λ_start[ci_in]
        iszero(mass) && continue

        chosen_action = actions[policy[ci_in]]
        next_axis_i   = stage.next_state_idx(cell, chosen_action)
        out_idxs      = Base.setindex(in_idxs, next_axis_i, choice_dim)
        Λ_end[CartesianIndex(out_idxs)] += mass
    end
    return Λ_end
end
