"""
Layout-changing stage that drops one axis. Backward broadcasts
`V_end` along the dropped axis; forward sums `Λ_start` along it. K
is the sum-along-axis operator (no V/θ dependence; no kernel).
"""
struct ForgetfulSumStageSpec <: AbstractStageSpec
    forget_axis :: Symbol
end

ForgetfulSumStageSpec(; forget_axis::Symbol) = ForgetfulSumStageSpec(forget_axis)

output_layout(spec::ForgetfulSumStageSpec, layout::StateLayout) =
    drop_axis(layout, spec.forget_axis)

# Backward / forward #
#--------------------#

function backward!(buffer, spec::ForgetfulSumStageSpec, V_end, env)
    @assert size(V_end) == layout_size(buffer.output_layout)
    buffer.V_start .= with_singleton(V_end, buffer.input_layout, spec.forget_axis)
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

function forward!(buffer, spec::ForgetfulSumStageSpec, Λ_start)
    @assert size(Λ_start) == layout_size(buffer.input_layout)
    sum!(with_singleton(buffer.Λ_end, buffer.input_layout, spec.forget_axis), Λ_start)
    return buffer.Λ_end
end

# Wrapper #
#---------#

@definestage ForgetfulSumStage ForgetfulSumStageSpec
