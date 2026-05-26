"""
Composition of stages. Pure data: a flat tuple of component Specs
and a mutable moments dict. Layout flows in at allocate time and
chains through the components.

The `moments` slot is the only mutable field on any Spec —
`define_moment!(chain, name, spec)` extends it.
"""
struct ChainStageSpec{Stages<:Tuple} <: AbstractStageSpec
    stages  :: Stages
    moments :: Dict{Symbol, Any}
end

function ChainStageSpec(stages::Tuple;
                        moments::Dict{Symbol, Any}=Dict{Symbol, Any}())
    @assert !isempty(stages)
    flat = _flatten_chain_specs(stages)
    return ChainStageSpec{typeof(flat)}(flat, moments)
end

"""
Flatten any nested `ChainStageSpec` components in a tuple. Single-level
walk; relies on `ChainStageSpec` already being flat.
"""
function _flatten_chain_specs(stages::Tuple)
    out = AbstractStageSpec[]
    for s in stages
        @assert s isa AbstractStageSpec
        s isa ChainStageSpec ? append!(out, s.stages) : push!(out, s)
    end
    return Tuple(out)
end

"""
Chain buffer: a tuple of per-component sub-buffers plus the chain's
own layout fields and cache. Each sub-buffer carries its own layout;
the chain's `input_layout` mirrors the first sub-buffer, `output_layout`
the last.
"""
struct ChainStageBuffer{Buffers<:Tuple, LIn, LOut} <: AbstractStageBuffer
    stages        :: Buffers
    input_layout  :: LIn
    output_layout :: LOut
    cache         :: CacheState
end

"Chain stage: a tuple of bundled sub-stages composed via `∘`."
struct ChainStage{Spec<:ChainStageSpec, Buffer<:ChainStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function ChainStage(stages::Tuple)
    @assert !isempty(stages)
    if all(s -> s isa AbstractStage, stages)
        specs, sub_buffers = _flatten_chain_stage_pairs(stages)
        spec = ChainStageSpec(specs)
        buf  = ChainStageBuffer(sub_buffers,
                                sub_buffers[1].input_layout,
                                sub_buffers[end].output_layout,
                                CacheState())
        return ChainStage{typeof(spec), typeof(buf)}(spec, buf)
    elseif all(s -> s isa AbstractStageSpec, stages)
        return ChainStage(ChainStageSpec(stages))
    else
        error("ChainStage components must be uniformly AbstractStage or AbstractStageSpec")
    end
end

"""
Flatten a tuple of bundled stages into parallel spec/buffer tuples,
unpacking any nested `ChainStage` so the Spec and Buffer sides stay
flat together. Preserves the original leaf buffers — no reallocation.
"""
function _flatten_chain_stage_pairs(stages::Tuple)
    specs = AbstractStageSpec[]
    bufs  = AbstractStageBuffer[]
    for s in stages
        if s isa ChainStage
            append!(specs, s.spec.stages)
            append!(bufs,  s.buffer.stages)
        else
            push!(specs, s.spec)
            push!(bufs,  s.buffer)
        end
    end
    return Tuple(specs), Tuple(bufs)
end

ChainStage(spec::ChainStageSpec) = error("ChainStage(spec) needs a layout; call ChainStage(spec, layout) or compose bundled stages with `∘`.")
ChainStage(spec::ChainStageSpec, layout::StateLayout) =
    ChainStage(spec, allocate(spec, layout))

bundle(spec::ChainStageSpec, layout::StateLayout) = ChainStage(spec, layout)
bundle(spec::ChainStageSpec, layout::StateLayout, ::Type{T}) where {T} =
    ChainStage(spec, allocate(spec, layout, T))

# Allocate — walk components, chaining layouts #
#----------------------------------------------#

function allocate(spec::ChainStageSpec, layout::StateLayout,
                  ::Type{T}=Float64) where {T}
    sub_buffers, out_layout = _allocate_chain_buffers(spec.stages, layout, T)
    return ChainStageBuffer(sub_buffers, layout, out_layout, CacheState())
end

"""
Sequentially allocate component buffers, threading each component's
`output_layout` into the next component's input. Returns the tuple
of sub-buffers and the chain's terminal output layout.
"""
function _allocate_chain_buffers(specs::Tuple, layout::StateLayout, T::Type)
    cur     = layout
    buffers = AbstractStageBuffer[]
    for s in specs
        b = allocate(s, cur, T)
        push!(buffers, b)
        cur = b.output_layout
    end
    return Tuple(buffers), cur
end

# Env slice — union over components #
#-----------------------------------#

"The set of `env` fields the chain's backward/forward needs (union over components)."
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
    missing_keys = Symbol[k for k in needed if !haskey(env, k)]
    isempty(missing_keys) ||
        error("env is missing required fields: $missing_keys; provided: $(keys(env))")
    return nothing
end

# Endpoint accessors #
#--------------------#

V_start_buffer(stage::ChainStage) = stage.buffer.stages[1].V_start
Λ_end_buffer(stage::ChainStage)   = stage.buffer.stages[end].Λ_end

# Backward sweep — type-stable via @generated #
#--------------------------------------------#
# Heterogeneous component tuple — a runtime `n:-1:1` loop on the
# tuple is type-unstable. `@generated` unrolls.

@generated function backward!(buffer::ChainStageBuffer{Buffers},
                              spec::ChainStageSpec{Stages},
                              V_end, env) where {Buffers<:Tuple, Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(V = backward!(buffer.stages[$i], spec.stages[$i], V, env))
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

@generated function forward!(buffer::ChainStageBuffer{Buffers},
                             spec::ChainStageSpec{Stages},
                             Λ_start) where {Buffers<:Tuple, Stages<:Tuple}
    N = length(Stages.parameters)
    calls = [:(Λ = forward!(buffer.stages[$i], spec.stages[$i], Λ))
             for i in 1:N]
    return quote
        Λ = Λ_start
        $(calls...)
        return Λ
    end
end

# Cache invalidation walks components #
#-------------------------------------#

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
this is the **opposite** of Julia's `∘` on `Function`. Auto-flattens
nested chains; refuses to compose chains that already carry moments.
"""
Base.:∘(a::AbstractStageSpec, b::AbstractStageSpec) =
    ChainStageSpec(_compose_spec_tuples(a, b))

Base.:∘(a::AbstractStage, b::AbstractStage) = ChainStage((a, b))

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
    @assert isempty(spec.moments) "∘: cannot compose a ChainStageSpec that already has moments on its $side; call define_moment! last."

# Legacy alias from pre-2026-05-19 — retired in a future cycle.
const ∘ₛ = Base.:∘
