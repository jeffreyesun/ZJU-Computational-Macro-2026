"""
Logit-smoothed discrete choice over a categorical axis. Actions are the
levels of `choice_axis`. The K-operator is the stochastic kernel
`Σ_a π(a|s) δ_{σ(s,a)}` with `π(a|s) = softmax((u_a) / ε)`; the kernel
stores the probability tensor `P[s..., a]`.

`flow_payoff(cell, action; env)`, `next_state_idx(cell, action) -> Int`,
and `closure_deps` mirror the [`Argmax`](@ref) convention. `ε::Param{T}`
is the logit scale, either a literal or a Symbol-valued sweep key (see
[`Param`](@ref)). `-Inf` flow payoffs are treated as "unavailable
action": skipped in the log-sum-exp and assigned zero probability.
"""
struct LogitChoice{F, BF,
                   LIn<:StateLayout, LOut<:StateLayout,
                   N, D, T<:Real, AV<:AbstractArray{T,N}} <: AbstractStage
    choice_axis    :: Symbol
    choice_dim     :: Int
    flow_payoff    :: F
    next_state_idx :: BF
    ε              :: Param{T}
    closure_deps   :: NTuple{D, Symbol}
    input_layout   :: LIn
    output_layout  :: LOut
    V_start        :: AV
    Λ_end          :: AV
end

"Construct a [`LogitChoice`](@ref) stage on `layout`."
function LogitChoice(layout::StateLayout;
                     choice_axis::Symbol,
                     flow_payoff,
                     next_state_idx,
                     ε::Param{T} = Param(1.0),
                     closure_deps::NTuple{D, Symbol} = (),
                     element_type::Union{Type, Nothing} = nothing,
                     V_start::Union{Nothing, AbstractArray} = nothing,
                     Λ_end::Union{Nothing, AbstractArray}  = nothing) where {D, T<:Real}
    Tb = @something element_type T
    choice_dim = axis_position(layout, choice_axis)
    dims       = layout_size(layout)
    N          = length(dims)
    Vs         = @something V_start zeros(Tb, dims)
    Λe         = @something Λ_end   zeros(Tb, dims)
    @assert typeof(Vs) === typeof(Λe) "LogitChoice: V_start and Λ_end must have the same concrete array type"
    return LogitChoice{typeof(flow_payoff),
                       typeof(next_state_idx),
                       typeof(layout), typeof(layout),
                       N, D, Tb, typeof(Vs)}(
        choice_axis, choice_dim, flow_payoff, next_state_idx, ε, closure_deps,
        layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:LogitChoice}) = NamedTuple()

function allocate(stage::LogitChoice{F,BF,LIn,LOut,N,D,T},
                  ::Type{T2} = T) where {F,BF,LIn,LOut,N,D,T,T2}
    dims      = layout_size(stage.input_layout)
    n_actions = axissize(stage.input_layout.axes[stage.choice_dim])
    prob      = zeros(T2, dims..., n_actions)
    return ((choice_prob = prob,), nothing)
end

# Backward #
#----------#

function backward!(stage::LogitChoice{F,BF,LIn,LOut,N,D,T},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {F,BF,LIn,LOut,N,D,T}
    ε        = resolve(stage.ε, env)
    layout   = stage.input_layout
    cdim     = stage.choice_dim
    actions  = axisvalues(layout.axes[cdim])
    n_a      = length(actions)
    V_start  = stage.V_start
    prob     = kernel.choice_prob

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)

        max_u   = typemin(T)
        any_fin = false
        # Pass 1: max_u for numerical stability.
        for action in actions
            r = stage.flow_payoff(cell, action; env = env)
            if isfinite(r)
                any_fin = true
                next_axis_i = stage.next_state_idx(cell, action)
                out_idxs = Base.setindex(in_idxs, next_axis_i, cdim)
                u = r + V_end[CartesianIndex(out_idxs)]
                if u > max_u
                    max_u = u
                end
            end
        end
        any_fin || error("LogitChoice: no finite-payoff action at cell $cell")

        # Pass 2: weights / unnormalised probs.
        denom = zero(T)
        for (a_i, action) in pairs(actions)
            r = stage.flow_payoff(cell, action; env = env)
            if isfinite(r)
                next_axis_i = stage.next_state_idx(cell, action)
                out_idxs    = Base.setindex(in_idxs, next_axis_i, cdim)
                u           = r + V_end[CartesianIndex(out_idxs)]
                w           = exp((u - max_u) / ε)
                prob[in_idxs..., a_i] = w
                denom += w
            else
                prob[in_idxs..., a_i] = zero(T)
            end
        end
        # Pass 3: normalise.
        for a_i in 1:n_a
            prob[in_idxs..., a_i] /= denom
        end
        V_start[ci_in] = max_u + ε * log(denom)
    end
    return V_start
end

# Forward #
#---------#

function forward!(stage::LogitChoice{F,BF,LIn,LOut,N,D,T},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {F,BF,LIn,LOut,N,D,T}
    layout  = stage.input_layout
    cdim    = stage.choice_dim
    actions = axisvalues(layout.axes[cdim])
    Λ_end   = stage.Λ_end
    prob    = kernel.choice_prob

    fill!(Λ_end, zero(T))
    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        mass    = Λ_start[ci_in]
        iszero(mass) && continue

        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = stage.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, cdim)
            Λ_end[CartesianIndex(out_idxs)] += mass * p
        end
    end
    return Λ_end
end
