"""
Logit-smoothed discrete-choice stage. Actions are levels of
`choice_axis`. K is the stochastic kernel `Σ_a π(a|s) δ_{σ(s,a)}`
with `π(a|s) = softmax(u_a / ε)`. `-Inf` flow payoffs are skipped
(assigned probability zero).
"""
struct LogitChoiceStageSpec{F, BF, T} <: AbstractStageSpec
    choice_axis    :: Symbol
    flow_payoff    :: F
    next_state_idx :: BF
    ε              :: T
end

function LogitChoiceStageSpec(; choice_axis::Symbol, flow_payoff, next_state_idx, ε=1.0)
    return LogitChoiceStageSpec(choice_axis, flow_payoff, next_state_idx, ε)
end

"Kernel: per-cell action-probability tensor and a cached map of next-state Cartesian indices, both of shape `(layout_size..., n_actions)`."
struct LogitChoiceKernel{P<:AbstractArray, I<:AbstractArray}
    choice_prob :: P
    next_ci     :: I
end

function allocate_kernel(spec::LogitChoiceStageSpec, ::Type{T}, layout::StateLayout) where {T}
    actions = axisvalues(layout.axes[axis_dim(layout, spec.choice_axis)])
    sz      = layout_size(layout)
    next_ci = reshape(
        [set_coord(CartesianIndex(Tuple(idx)), layout, spec.choice_axis => spec.next_state_idx(cell, action))
         for (idx, cell) in cells(layout), action in actions],
        sz..., length(actions),
    )
    return LogitChoiceKernel(zeros(T, sz..., length(actions)), next_ci)
end

# Backward / forward #
#--------------------#

function backward!(buffer, spec::LogitChoiceStageSpec, V_end, env)
    (;V_start, input_layout, ε) = resolve(buffer, spec, env)
    (;choice_prob, next_ci) = buffer.kernel
    actions = axisvalues(input_layout.axes[axis_position(input_layout, spec.choice_axis)])
    T       = eltype(V_start)

    # Gather U[s, a] = r(s, a) + V_end[σ(s, a)] into choice_prob (used as scratch).
    for (idx, cell) in cells(input_layout), (a_i, action) in pairs(actions)
        r = spec.flow_payoff(cell, action; env=env)
        in_idxs = Tuple(idx)
        choice_prob[in_idxs..., a_i] = isfinite(r) ? r + V_end[next_ci[in_idxs..., a_i]] : typemin(T)
    end

    _softmax_and_lse_along_last!(V_start, choice_prob, ε)
    _seat_cache!(buffer, V_end, env)
    return V_start
end

function forward!(buffer, spec::LogitChoiceStageSpec, Λ_start)
    (;Λ_end) = resolve(buffer, spec)
    (;choice_prob, next_ci) = buffer.kernel
    n_a = last(size(choice_prob))

    fill!(Λ_end, zero(eltype(Λ_end)))
    for ci_in in CartesianIndices(Λ_start)
        mass = Λ_start[ci_in]
        iszero(mass) && continue
        in_idxs = Tuple(ci_in)
        for a_i in 1:n_a
            p = choice_prob[in_idxs..., a_i]
            iszero(p) && continue
            Λ_end[next_ci[in_idxs..., a_i]] += mass * p
        end
    end
    return Λ_end
end

# Wrapper #
#---------#

@definestage LogitChoiceStage LogitChoiceStageSpec kernel=LogitChoiceKernel
