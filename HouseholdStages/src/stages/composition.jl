"""
Configuration for a composition of stages. Pure data: a tuple of
component Specs, a mutable moments dict, and the output layout
derived from the last component.

Constructed via `ChainStageSpec((s1, s2, …))` or
`spec1 ∘ spec2 ∘ …`. Auto-flattens nested chains so the resulting
`stages` tuple is always flat.

The `moments` field is intentionally a `Dict{Symbol, MomentSpec}` —
the only mutable slot on any Spec — so `define_moment!(chain, name,
spec)` can append.
"""
struct ChainStageSpec{Stages<:Tuple, L} <: AbstractStageSpec
    stages     :: Stages
    moments    :: Dict{Symbol, Any}
    out_layout :: L
end

function ChainStageSpec(stages::Tuple;
                        moments::Dict{Symbol, Any} = Dict{Symbol, Any}())
    isempty(stages) && error("ChainStageSpec must contain at least one stage")
    # Refuse nested ChainStageSpec: auto-flatten in `∘`.
    flat = _flatten_chain_specs(stages)
    out  = _spec_out_layout(flat[end])
    return ChainStageSpec{typeof(flat), typeof(out)}(flat, moments, out)
end

"""CLAUDE
Flatten any nested ChainStageSpec components in a tuple. `((a, b),
c, (d,))` returns `(a, b, c, d)`. Single-level — relies on the
invariant that ChainStageSpec is itself already flat.
"""
function _flatten_chain_specs(stages::Tuple)
    out = AbstractStageSpec[]
    for s in stages
        s isa AbstractStageSpec || error("ChainStageSpec component must be an AbstractStageSpec, got $(typeof(s))")
        if s isa ChainStageSpec
            append!(out, s.stages)
        else
            push!(out, s)
        end
    end
    return Tuple(out)
end

"""CLAUDE
Output layout of a Spec. Looks up the `output_layout` field; every
package-level Spec has one. Falls back to an explicit error so
mis-typed Specs surface early.
""" #CLAUDE You don't need this. You can just call `spec.output_layout` directly and let the error happen naturally if the field is missing.
_spec_out_layout(spec::AbstractStageSpec) =
    hasfield(typeof(spec), :output_layout) ? spec.output_layout :
        error("Spec of type $(typeof(spec)) has no output_layout field; " *
              "cannot derive chain output layout")

"""
Per-call buffer for a chain. Holds a tuple of per-component Buffers
(`stages[i]` is the buffer for `chain.spec.stages[i]`) plus the
chain-level cache. The chain's cache mirrors the terminal-stage
buffer's `(V_end, env)` so the generic cache-checking
[`forward!`](@ref) on `AbstractStage` works uniformly.
"""
struct ChainStageBuffer{Buffers<:Tuple} <: AbstractStageBuffer
    stages :: Buffers
    cache  :: CacheState
end

"""
A composition of stages. Construct as
`s1 ∘ s2 ∘ s3` or `ChainStage((s1, s2, s3))`. The `∘` operator
composes left-to-right in *time* (`s1` runs first; note this is the
**opposite** of `Base.∘` on `Function`).

Attach moments via [`define_moment!`](@ref) / [`define_moments!`](@ref);
read them via [`compute_moments`](@ref).
"""
struct ChainStage{Spec<:ChainStageSpec, Buffer<:ChainStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function ChainStage(stages::Tuple)
    isempty(stages) && error("ChainStage must contain at least one stage")
    # Accept a tuple of Stages or a tuple of Specs.
    if all(s -> s isa AbstractStage, stages)
        spec_tuple = map(s -> s.spec, stages)
        spec       = ChainStageSpec(spec_tuple)
        return ChainStage(spec, allocate(spec))
    elseif all(s -> s isa AbstractStageSpec, stages)
        spec = ChainStageSpec(stages)
        return ChainStage(spec, allocate(spec))
    else
        error("ChainStage components must be uniformly AbstractStage or AbstractStageSpec")
    end
end

ChainStage(spec::ChainStageSpec) = ChainStage(spec, allocate(spec))
bundle(spec::ChainStageSpec)     = ChainStage(spec)

# Allocate #
#----------#

# Default: each component picks its own `spec.element_type` (so a chain
# whose components were lifted to Dual keeps Dual buffers throughout).
# Explicit `allocate(spec, T)` overrides every component to T.
function allocate(spec::ChainStageSpec)
    component_buffers = map(allocate, spec.stages)
    return ChainStageBuffer{typeof(component_buffers)}(
        component_buffers, CacheState(),
    )
end

function allocate(spec::ChainStageSpec, ::Type{T}) where {T}
    component_buffers = map(s -> allocate(s, T), spec.stages)
    return ChainStageBuffer{typeof(component_buffers)}(
        component_buffers, CacheState(),
    )
end

# Env slice — union over components #
#-----------------------------------#

"""
    chain_env_names(spec_or_chain) -> NTuple{N, Symbol}

The union of `effective_env_slice` across all stages in the chain —
the set of `env` fields the chain's backward / forward needs.
"""
function chain_env_names(spec::ChainStageSpec)
    names = Symbol[]
    for s in spec.stages
        for k in effective_env_slice(s)
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end
chain_env_names(chain::ChainStage) = chain_env_names(chain.spec)

effective_env_slice(spec::ChainStageSpec) = chain_env_names(spec)

function validate_env(spec::ChainStageSpec, env)
    needed = chain_env_names(spec)
    missing_keys = Symbol[]
    for k in needed
        haskey(env, k) || push!(missing_keys, k)
    end
    isempty(missing_keys) ||
        error("env is missing required fields: $(missing_keys); provided keys: $(keys(env))")
    return nothing
end

# V_start / Λ_end accessors — borrow from endpoint buffers #
#---------------------------------------------------------#

V_start_buffer(stage::ChainStage) = stage.buffer.stages[1].V_start
Λ_end_buffer(stage::ChainStage)   = stage.buffer.stages[end].Λ_end

# Backward sweep — type-stable via @generated #
#--------------------------------------------#
#
# A runtime `for i in n:-1:1` over `c.spec.stages[i]` is type-unstable
# because the tuple is heterogeneous (different concrete Spec types per
# slot). `@generated` unrolls the tuple statically and recovers full
# type stability — ~40 alloc / pass disappear and the chain
# backward+forward sits within ~1.0x of the hand-coded reference.

@generated function backward!(spec::ChainStageSpec{Stages}, V_end, env,
                              buffer::ChainStageBuffer) where {Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(V = backward!(spec.stages[$i], V, env, buffer.stages[$i]))
             for i in N:-1:1]
    return quote
        V = V_end
        $(calls...)
        _seat_cache!(buffer, V_end, env)
        return V
    end
end

# Forward sweep #
#---------------#

@generated function forward!(spec::ChainStageSpec{Stages}, Λ_start,
                              buffer::ChainStageBuffer) where {Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(Λ = forward!(spec.stages[$i], Λ, buffer.stages[$i]))
             for i in 1:N]
    return quote
        Λ = Λ_start
        $(calls...)
        return Λ
    end
end

# Cache invalidation walks components too #
#-----------------------------------------#

function invalidate!(buffer::ChainStageBuffer)
    buffer.cache.kernel_valid = false
    for b in buffer.stages
        invalidate!(b)
    end
    return buffer
end

# Composition operators #
#-----------------------#

"""
    s1 ∘ s2

Left-to-right (time-ordered) stage composition. `s1` runs first;
this is the **opposite** of Julia's `∘` on `Function` (rightmost
runs first).

Defined on `AbstractStageSpec` (the primary, no-allocation form)
and on `AbstractStage` (the sugar that bundles a fresh chain
buffer). Auto-flattens nested chains and refuses to compose chains
that already carry moments (call `define_moment!` last).
"""
Base.:∘(a::AbstractStageSpec, b::AbstractStageSpec) =
    ChainStageSpec(_compose_spec_tuples(a, b))

Base.:∘(a::AbstractStage, b::AbstractStage) =
    bundle(a.spec ∘ b.spec)

function _compose_spec_tuples(a::AbstractStageSpec, b::AbstractStageSpec)
    if a isa ChainStageSpec
        _assert_no_moments(a, "left")
        if b isa ChainStageSpec
            _assert_no_moments(b, "right")
            return (a.stages..., b.stages...)
        else
            return (a.stages..., b)
        end
    elseif b isa ChainStageSpec
        _assert_no_moments(b, "right")
        return (a, b.stages...)
    else
        return (a, b)
    end
end

_assert_no_moments(spec::ChainStageSpec, side::AbstractString) =
    isempty(spec.moments) ||
        error("∘: cannot compose a ChainStageSpec that already has moments on its $side. " *
              "Call `define_moment!` (or `define_moments!`) last, after all composition.")

# Legacy alias: `∘ₛ` was the operator before the 2026-05-19 refactor.
# Kept as a deprecation alias for now; can be retired after one cycle.
const ∘ₛ = Base.:∘
