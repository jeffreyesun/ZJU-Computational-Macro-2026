"""
Logit-smoothed discrete choice over a categorical axis. Actions are the
levels of `choice_axis`. The K-operator is the stochastic kernel
`Σ_a π(a|s) δ_{σ(s,a)}` with `π(a|s) = softmax((u_a) / ε)`; the kernel
stores the probability tensor `P[s..., a]`.

`flow_payoff(cell, action; env)` and `next_state_idx(cell, action) -> Int`
mirror the [`ArgmaxStage`](@ref) convention. `ε::Param{T}` is the logit
scale, either a literal or a Symbol-valued sweep key (see [`Param`](@ref)).
`-Inf` flow payoffs are treated as "unavailable action": skipped in the
log-sum-exp and assigned zero probability.
"""
struct LogitChoiceStage{F, BF,
                   LIn<:StateLayout, LOut<:StateLayout,
                   N, T<:Real, AV<:AbstractArray{T,N}} <: AbstractStage
    choice_axis    :: Symbol
    choice_dim     :: Int
    flow_payoff    :: F
    next_state_idx :: BF
    ε              :: Param{T}
    input_layout   :: LIn
    output_layout  :: LOut
    V_start        :: AV
    Λ_end          :: AV
end

"Construct a [`LogitChoiceStage`](@ref) stage on `layout`."
function LogitChoiceStage(layout::StateLayout;
                     choice_axis::Symbol,
                     flow_payoff,
                     next_state_idx,
                     ε::Param{T} = Param(1.0),
                     element_type::Union{Type, Nothing} = nothing,
                     V_start::Union{Nothing, AbstractArray} = nothing,
                     Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    Tb = @something element_type T
    choice_dim = axis_position(layout, choice_axis)
    (; Vs, Λe) = _alloc_VΛ(layout, Tb, V_start, Λ_end)
    return LogitChoiceStage{typeof(flow_payoff),
                       typeof(next_state_idx),
                       typeof(layout), typeof(layout),
                       ndims(Vs), Tb, typeof(Vs)}(
        choice_axis, choice_dim, flow_payoff, next_state_idx, ε,
        layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:LogitChoiceStage}) = NamedTuple()

function allocate(stage::LogitChoiceStage, ::Type{T2} = eltype(stage.V_start)) where {T2}
    dims      = layout_size(stage.input_layout)
    n_actions = axissize(stage.input_layout.axes[stage.choice_dim])
    prob      = zeros(T2, dims..., n_actions)
    return (; kernel = (; choice_prob = prob), scratch = nothing)
end

# Backward #
#----------#

function backward!(stage::LogitChoiceStage, V_end, env, buffers)
    (; kernel, scratch) = buffers
    (; input_layout, choice_dim, V_start) = stage
    ε       = resolve(stage.ε, env)
    actions = axisvalues(input_layout.axes[choice_dim])
    n_a     = length(actions)
    prob    = kernel.choice_prob
    T       = eltype(V_start)

    for (idx, cell) in cells(input_layout)
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
                out_idxs = Base.setindex(in_idxs, next_axis_i, choice_dim)
                u = r + V_end[CartesianIndex(out_idxs)]
                if u > max_u
                    max_u = u
                end
            end
        end
        any_fin || error("LogitChoiceStage: no finite-payoff action at cell $cell")

        # Pass 2: weights / unnormalised probs.
        denom = zero(T)
        for (a_i, action) in pairs(actions)
            r = stage.flow_payoff(cell, action; env = env)
            if isfinite(r)
                next_axis_i = stage.next_state_idx(cell, action)
                out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
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

function forward!(stage::LogitChoiceStage, Λ_start, buffers, moments = nothing)
    (; kernel, scratch) = buffers
    (; input_layout, choice_dim, Λ_end) = stage
    actions = axisvalues(input_layout.axes[choice_dim])
    prob    = kernel.choice_prob
    T       = eltype(Λ_end)

    fill!(Λ_end, zero(T))
    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        mass    = Λ_start[ci_in]
        iszero(mass) && continue

        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = stage.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            Λ_end[CartesianIndex(out_idxs)] += mass * p
        end
    end
    return Λ_end
end
