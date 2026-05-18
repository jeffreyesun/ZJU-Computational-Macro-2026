"""
A composition of stages. Backward walks the chain from end to start;
forward walks it from start to end. The chain itself is a stage, so
chains-of-chains are well-defined and stages form a category under
composition. The chain's `V_start` is the first inner stage's `V_start`,
its `Λ_end` is the last inner stage's `Λ_end`.

Construct as `StageChain((s1, s2, s3))` or `s1 ∘ₛ s2 ∘ₛ s3`. The `∘ₛ`
operator (Greek `\\circ` + `s`) composes left-to-right in *time* (`s1`
runs first).
"""
struct StageChain{Stages<:Tuple} <: AbstractStage
    stages::Stages
    function StageChain{Stages}(stages::Stages) where {Stages<:Tuple}
        isempty(stages) && error("StageChain must contain at least one stage")
        return new{Stages}(stages)
    end
end

StageChain(stages::Stages) where {Stages<:Tuple} =
    StageChain{Stages}(stages)

"""
    chain_env_names(chain::StageChain) -> NTuple{N, Symbol}

The union of `effective_env_slice` across all stages in the chain — the
set of `env` fields the chain's backward / forward needs.
"""
function chain_env_names(chain::StageChain)
    names = Symbol[]
    for s in chain.stages
        for k in effective_env_slice(s)
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end

effective_env_slice(chain::StageChain) = chain_env_names(chain)

function validate_env(chain::StageChain, env)
    needed = chain_env_names(chain)
    missing_keys = Symbol[]
    for k in needed
        haskey(env, k) || push!(missing_keys, k)
    end
    isempty(missing_keys) ||
        error("env is missing required fields: $(missing_keys); provided keys: $(keys(env))")
    return nothing
end

# Buffer accessors override the defaults: a chain doesn't store its own
# V_start / Λ_end; it borrows from its endpoints.
V_start_buffer(c::StageChain) = V_start_buffer(first(c.stages))
Λ_end_buffer(c::StageChain) = Λ_end_buffer(last(c.stages))

# Allocate #
#----------#

# A chain's workspace is a tuple of per-stage workspaces. Each element of
# `kernels` / `scratches` is the corresponding stage's `kernel` /
# `scratch`.
function allocate(c::StageChain, ::Type{T} = Float64) where {T}
    pairs    = map(s -> allocate(s, T), c.stages)
    kernels   = map(first, pairs)
    scratches = map(last, pairs)
    return (kernels, scratches)
end

# Backward pass #
#---------------#
#
# Implementation note: a runtime `for i in n:-1:1` loop over `c.stages[i]`
# is type-unstable because `c.stages` is a heterogeneous `Tuple` (each
# element a different concrete stage type). The compiler produces a
# `Union{...}` lowering of every per-iteration call, with small dynamic
# boxes per chain pass. Replacing the loop with a `@generated` body
# unrolls the tuple statically and recovers full type stability — about
# 40 alloc / pass disappear and the chain backward+forward drops from
# 1.03x → ~1.00x of the hand-coded reference.

@generated function backward!(c::StageChain{Stages}, V_end, env,
                               kernels::Tuple, scratches::Tuple) where {Stages<:Tuple}
    N = length(Stages.parameters)
    body = Expr(:block)
    push!(body.args, :(V = V_end))
    for i in N:-1:1
        push!(body.args, :(V = backward!(c.stages[$i], V, env,
                                          kernels[$i], scratches[$i])))
    end
    push!(body.args, :(return V))
    return body
end

# Forward pass #
#--------------#

@generated function forward!(c::StageChain{Stages}, Λ_start,
                              kernels::Tuple, scratches::Tuple,
                              moments = nothing) where {Stages<:Tuple}
    N = length(Stages.parameters)
    body = Expr(:block)
    push!(body.args, :(Λ = Λ_start))
    for i in 1:N
        push!(body.args, :(Λ = forward!(c.stages[$i], Λ,
                                         kernels[$i], scratches[$i], moments)))
    end
    push!(body.args, :(return Λ))
    return body
end

# Composition operator #
#----------------------#

"""
    s1 ∘ₛ s2

Left-to-right stage composition: in the resulting chain, `s1`'s forward
pass runs first, then `s2`'s. Semantically, this is the *time* order of
the stages within a period.

`∘ₛ` is associative and produces a flat `StageChain` regardless of how
parentheses are placed.
"""
∘ₛ(a::AbstractStage, b::AbstractStage) = StageChain((a, b))
∘ₛ(a::StageChain, b::AbstractStage) = StageChain((a.stages..., b))
∘ₛ(a::AbstractStage, b::StageChain) = StageChain((a, b.stages...))
∘ₛ(a::StageChain, b::StageChain) = StageChain((a.stages..., b.stages...))
