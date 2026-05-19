"""
A composition of stages, optionally carrying moment specs.

Construct as `ChainStage((s1, s2, s3))` or `s1 ∘ₛ s2 ∘ₛ s3`. The `∘ₛ`
operator (Greek `\\circ` + `s`) composes left-to-right in *time* (`s1`
runs first). Moments are attached via [`lift_moments`](@ref), which
returns a new `ChainStage` with the `moments` field populated (it is an
empty `NamedTuple` by default). Reading the moments after a forward
pass is the job of [`compute_moments`](@ref).
"""
struct ChainStage{Stages<:Tuple, Specs<:NamedTuple, L} <: AbstractStage
    stages     :: Stages
    moments    :: Specs
    out_layout :: L
end

# Outer constructor. `out_layout` is the last stage's output layout; it
# sits unused until lift_moments lights up compute_moments, but it's
# cheap to derive and keeps the struct shape uniform across the
# moments-empty and moments-non-empty cases.
function ChainStage(stages::Tuple; moments::NamedTuple = (;))
    isempty(stages) && error("ChainStage must contain at least one stage")
    out = _stage_out_layout(stages[end])
    return ChainStage{typeof(stages), typeof(moments), typeof(out)}(stages, moments, out)
end

"""CLAUDE
Output layout of the terminal stage. Falls back to an explicit error
when the stage does not expose an `output_layout` field — every
package-level stage does, so this is informational rather than
operational.
"""
_stage_out_layout(stage::AbstractStage) =
    hasfield(typeof(stage), :output_layout) ? stage.output_layout :
        error("stage of type $(typeof(stage)) has no output_layout field; cannot derive chain output layout")

"""
    chain_env_names(chain::ChainStage) -> NTuple{N, Symbol}

The union of `effective_env_slice` across all stages in the chain — the
set of `env` fields the chain's backward / forward needs.
"""
function chain_env_names(chain::ChainStage)
    names = Symbol[]
    for s in chain.stages
        for k in effective_env_slice(s)
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end

effective_env_slice(chain::ChainStage) = chain_env_names(chain)

function validate_env(chain::ChainStage, env)
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
V_start_buffer(c::ChainStage) = V_start_buffer(first(c.stages))
Λ_end_buffer(c::ChainStage) = Λ_end_buffer(last(c.stages))

# Allocate #
#----------#

# A chain's workspace is a `Tuple` of per-stage `(; kernel, scratch)` bundles
# — element `i` is the bundle for `c.stages[i]`. backward!/forward! pass
# `buffers[i]` through to each stage without unpacking.
"""
    allocate(chain::ChainStage, T = Float64) -> Tuple

Build the per-stage workspace tuple for a chain. Element `i` is the
bundle returned by `allocate(c.stages[i], T)` — a NamedTuple
`(; kernel, scratch)`. Pass the whole tuple to `backward!` / `forward!`;
the chain forwards `buffers[i]` to each stage.
"""
function allocate(c::ChainStage, ::Type{T} = Float64) where {T}
    return ntuple(i -> allocate(c.stages[i], T), length(c.stages))
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

@generated function backward!(c::ChainStage{Stages}, V_end, env, buffers) where {Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(V = backward!(c.stages[$i], V, env, buffers[$i]))
             for i in N:-1:1]
    return quote
        V = V_end
        $(calls...)
        return V
    end
end

# Forward pass #
#--------------#

@generated function forward!(c::ChainStage{Stages}, Λ_start, buffers,
                              moments = nothing) where {Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(Λ = forward!(c.stages[$i], Λ, buffers[$i], moments))
             for i in 1:N]
    return quote
        Λ = Λ_start
        $(calls...)
        return Λ
    end
end

# Composition operator #
#----------------------#

"""
    s1 ∘ₛ s2

Left-to-right stage composition: in the resulting chain, `s1`'s forward
pass runs first, then `s2`'s. Semantically, this is the *time* order of
the stages within a period.

`∘ₛ` is associative and produces a flat `ChainStage` regardless of how
parentheses are placed. Composing a chain that already carries moments
errors — call `lift_moments` last, after composition is finalised.
"""
∘ₛ(a::AbstractStage, b::AbstractStage) = ChainStage((a, b))

function ∘ₛ(a::ChainStage, b::AbstractStage)
    _assert_no_moments(a, "left")
    return ChainStage((a.stages..., b))
end

function ∘ₛ(a::AbstractStage, b::ChainStage)
    _assert_no_moments(b, "right")
    return ChainStage((a, b.stages...))
end

function ∘ₛ(a::ChainStage, b::ChainStage)
    _assert_no_moments(a, "left")
    _assert_no_moments(b, "right")
    return ChainStage((a.stages..., b.stages...))
end

_assert_no_moments(c::ChainStage, side::AbstractString) =
    isempty(c.moments) ||
        error("∘ₛ: cannot compose a ChainStage that already has moments on its $side. " *
              "Call `lift_moments` last, after all `∘ₛ` composition.")
