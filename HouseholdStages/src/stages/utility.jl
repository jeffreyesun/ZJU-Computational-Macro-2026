"""
State-only flow-utility stage: `V_start[s] = u(cell; env) + V_end[s]`,
identity on Λ. `utility` is a `(cell; env)` closure.
"""
struct UtilityStageSpec{F} <: AbstractStageSpec
    utility :: F
end

UtilityStageSpec(; utility) = UtilityStageSpec{typeof(utility)}(utility)

# Backward / forward #
#--------------------#

function backward!(buffer, spec::UtilityStageSpec, V_end, env)
    buffer.V_start .= spec.utility.(cell_array(buffer.input_layout); env) .+ V_end
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

function forward!(buffer, spec::UtilityStageSpec, Λ_start)
    copyto!(buffer.Λ_end, Λ_start)
    return buffer.Λ_end
end

# Wrapper #
#---------#

@definestage UtilityStage UtilityStageSpec
