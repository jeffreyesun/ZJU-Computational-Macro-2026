"""
Hard discrete choice over a categorical axis. Actions are the levels of
`choice_axis`. The K-operator is a sparse permutation (`δ_{π(s)}` for
each input cell, where `π(s)` is the chosen-action's next-cell index);
the kernel stores the integer policy.

`flow_payoff(cell, action; env)` is the period payoff at the cell of
taking action `action`; `-Inf` is treated as "unavailable" and skipped.
`next_state_idx(cell, action) -> Int` returns the integer index along
`choice_axis` of the cell reached by `action`. `closure_deps` lists env
fields the two closures read.
"""
struct Argmax{F, BF,
              LIn<:StateLayout, LOut<:StateLayout,
              N, D, T<:Real, AV<:AbstractArray{T,N}} <: AbstractStage
    choice_axis    :: Symbol
    choice_dim     :: Int
    flow_payoff    :: F
    next_state_idx :: BF
    closure_deps   :: NTuple{D, Symbol}
    input_layout   :: LIn
    output_layout  :: LOut
    V_start        :: AV
    Λ_end          :: AV
    policy         :: Array{Int, N}
end

"""
    Argmax(layout; choice_axis, flow_payoff, next_state_idx,
                  closure_deps=(), element_type=Float64) -> Argmax

Construct an [`Argmax`](@ref) stage. `flow_payoff(cell, action; env)`
and `next_state_idx(cell, action)` are user closures (see the struct
docstring).
"""
function Argmax(layout::StateLayout;
                choice_axis::Symbol,
                flow_payoff,
                next_state_idx,
                closure_deps::NTuple{D, Symbol} = (),
                element_type::Type{T} = Float64,
                V_start::Union{Nothing, AbstractArray} = nothing,
                Λ_end::Union{Nothing, AbstractArray}  = nothing) where {D, T<:Real}
    choice_dim = axis_position(layout, choice_axis)
    dims       = layout_size(layout)
    N          = length(dims)
    Vs         = @something V_start zeros(T, dims)
    Λe         = @something Λ_end   zeros(T, dims)
    @assert typeof(Vs) === typeof(Λe) "Argmax: V_start and Λ_end must have the same concrete array type"
    policy     = zeros(Int, dims)
    return Argmax{typeof(flow_payoff),
                  typeof(next_state_idx),
                  typeof(layout), typeof(layout),
                  N, D, T, typeof(Vs)}(
        choice_axis, choice_dim, flow_payoff, next_state_idx, closure_deps,
        layout, layout, Vs, Λe, policy,
    )
end

static_env_deps(::Type{<:Argmax}) = NamedTuple()

function allocate(stage::Argmax{F,BF,LIn,LOut,N,D,T},
                  ::Type{T2} = T) where {F,BF,LIn,LOut,N,D,T,T2}
    return ((policy = stage.policy,), nothing)
end

# Backward #
#----------#

function backward!(stage::Argmax{F,BF,LIn,LOut,N,D,T},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {F,BF,LIn,LOut,N,D,T}
    layout     = stage.input_layout
    choice_dim = stage.choice_dim
    actions    = axisvalues(layout.axes[choice_dim])
    V_start    = stage.V_start
    policy     = stage.policy

    for (idx, cell) in cells(layout)
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
        best_a_i == 0 && error("Argmax: no finite-payoff action at cell $cell")

        V_start[ci_in] = best_v
        policy[ci_in]  = best_a_i
    end
    return V_start
end

# Forward #
#---------#

function forward!(stage::Argmax{F,BF,LIn,LOut,N,D,T},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {F,BF,LIn,LOut,N,D,T}
    layout     = stage.input_layout
    choice_dim = stage.choice_dim
    actions    = axisvalues(layout.axes[choice_dim])
    Λ_end      = stage.Λ_end
    policy     = stage.policy

    fill!(Λ_end, zero(T))
    for (idx, cell) in cells(layout)
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
