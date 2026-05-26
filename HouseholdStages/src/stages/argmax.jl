"""
Hard discrete-choice stage. Actions are the levels of `choice_axis`.
K is a sparse permutation (`δ_{π(s)}` for each input cell); the
materialised integer policy lives on the kernel.

  - `flow_payoff(cell, action; env)` — period payoff; `-Inf` means "unavailable".
  - `next_state_idx(cell, action) -> Int` — index along `choice_axis` of the
    cell reached by the action.
"""
struct ArgmaxStageSpec{F, BF} <: AbstractStageSpec
    choice_axis    :: Symbol
    flow_payoff    :: F
    next_state_idx :: BF
end

ArgmaxStageSpec(; choice_axis, flow_payoff, next_state_idx) =
    ArgmaxStageSpec{typeof(flow_payoff), typeof(next_state_idx)}(
        choice_axis, flow_payoff, next_state_idx,
    )

"Kernel: integer policy array and a cached map of next-state Cartesian indices, shape `(layout_size..., n_actions)`."
struct ArgmaxKernel{P<:AbstractArray{Int}, I<:AbstractArray}
    policy  :: P
    next_ci :: I
end

function allocate_kernel(spec::ArgmaxStageSpec, ::Type, layout::StateLayout)
    actions = axisvalues(layout.axes[axis_dim(layout, spec.choice_axis)])
    sz      = layout_size(layout)
    next_ci = reshape(
        [set_coord(CartesianIndex(Tuple(idx)), layout, spec.choice_axis => spec.next_state_idx(cell, action))
         for (idx, cell) in cells(layout), action in actions],
        sz..., length(actions),
    )
    return ArgmaxKernel(zeros(Int, sz), next_ci)
end

# Backward #
#----------#

function backward!(buffer, spec::ArgmaxStageSpec, V_end, env)
    (;V_start, kernel, input_layout) = buffer
    (;policy, next_ci) = kernel
    actions = axisvalues(input_layout.axes[axis_position(input_layout, spec.choice_axis)])
    T       = eltype(V_start)

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)

        best_v   = typemin(T)
        best_a_i = 0
        for (a_i, action) in pairs(actions)
            r = spec.flow_payoff(cell, action; env=env)
            isfinite(r) || continue
            v = r + V_end[next_ci[in_idxs..., a_i]]
            if v > best_v
                best_v   = v
                best_a_i = a_i
            end
        end
        @assert best_a_i > 0

        V_start[ci_in] = best_v
        policy[ci_in]  = best_a_i
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward #
#---------#

function forward!(buffer, spec::ArgmaxStageSpec, Λ_start)
    (;Λ_end, kernel) = buffer
    (;policy, next_ci) = kernel

    fill!(Λ_end, zero(eltype(Λ_end)))
    for ci_in in CartesianIndices(Λ_start)
        mass = Λ_start[ci_in]
        iszero(mass) && continue
        in_idxs = Tuple(ci_in)
        Λ_end[next_ci[in_idxs..., policy[ci_in]]] += mass
    end
    return Λ_end
end

# Wrapper #
#---------#

@definestage ArgmaxStage ArgmaxStageSpec kernel=ArgmaxKernel
