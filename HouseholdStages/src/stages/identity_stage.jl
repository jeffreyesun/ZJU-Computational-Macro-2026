"No-op stage whose K is the identity on `M(S)`. Useful as a `product` branch when one component performs no within-period action."
struct IdentityStageSpec <: AbstractStageSpec end

# Backward / forward #
#--------------------#

function backward!(buffer, spec::IdentityStageSpec, V_end, env)
    copyto!(buffer.V_start, V_end)
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

function forward!(buffer, spec::IdentityStageSpec, Λ_start)
    copyto!(buffer.Λ_end, Λ_start)
    return buffer.Λ_end
end

# Wrapper #
#---------#

@definestage IdentityStage IdentityStageSpec
