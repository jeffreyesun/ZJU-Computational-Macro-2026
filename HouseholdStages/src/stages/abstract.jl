"""
    AbstractStageSpec

The configuration layer of a stage: pure data describing what the
stage *is* (transition matrix, axis, layout, closures, `Param`-typed
hyperparameters). A Spec is immutable — except, for chains, the
moments slot, which is intentionally a `Dict` so `define_moment!` can
extend it.

Specs are internal types. Users see only the bundled
[`AbstractStage`](@ref). Each concrete stage class defines an
`<X>StageSpec` whose fields hold whatever the K-operator needs.
"""
abstract type AbstractStageSpec end

"""
    AbstractStageBuffer

The per-call state layer of a stage: the runtime data
`backward!` produces and `forward!` consumes. A Buffer holds

  * `kernel`  — the materialised K-operator data (`nothing` when K is
    config-only, e.g. Markov), populated by `backward!`.
  * `scratch` — any internal workspace that carries no morphism
    content (permuted-axis arrays, materialised cells, etc.).
  * `V_start` and `Λ_end` — the layout-shaped output buffers for
    backward and forward respectively.
  * `cache` — a [`CacheState`](@ref) tracking the most recent
    `(V_end, env)` seen by `backward!`, so that a later `forward!`
    can check whether its kernel is still valid.

Buffers are *fresh* on every `allocate(spec)` call. They are not
shared across different uses of the same Spec — in particular,
per-period buffers in a transition path are distinct Buffer
instances sharing one Spec.

A Buffer's `cache` field is its only mutable part.
"""
abstract type AbstractStageBuffer end

"""
    AbstractStage

The user-facing layer: a bundle of one [`AbstractStageSpec`](@ref)
and one [`AbstractStageBuffer`](@ref). Construct a `MarkovStage`,
`ArgmaxStage`, etc.; compose under `∘` and `×`; pass to
`backward!` / `forward!` / `solve_steady_state_given_env!` /
`solve_transition_given_env_path!`.

The legacy supertype name. The Spec/Buffer separation lives below;
this is the only level users see.
"""
abstract type AbstractStage end

"""CLAUDE
Mutable cache fingerprint for a stage Buffer. Tracks the `(V_end,
env)` last seen by `backward!` so that a later `forward!` can decide
whether its kernel is still valid.

`last_V_hash` is the full-array hash of V_end (catches in-place
mutation that pointer-equality would miss). `last_env` is the last
env NamedTuple, compared via `isequal`. `kernel_valid` is `true`
after a successful `backward!` and `false` after explicit
[`invalidate!`](@ref).
"""
mutable struct CacheState
    last_V_hash  :: UInt
    last_env     :: Any            # NamedTuple or nothing
    kernel_valid :: Bool
end
CacheState() = CacheState(zero(UInt), nothing, false)

"""CLAUDE
Stamp the buffer's cache with the `(V_end, env)` pair backward just
consumed. Call from concrete-stage `backward!` implementations
*after* the kernel has been populated.
"""
function _seat_cache!(buffer::AbstractStageBuffer, V_end, env)
    c = buffer.cache
    c.last_V_hash  = hash(V_end)
    c.last_env     = env
    c.kernel_valid = true
    return buffer
end

"""
    invalidate!(buffer_or_stage) -> buffer_or_stage

Mark the cache as stale. The next cache-checking `forward!` will
either error (default) or re-seat (if `reseat_if_stale=true`).

Useful for outer-loop drivers that mutate the chain's state in ways
the cache can't detect (e.g., a future eltype-switch lift). The
helpers `solve_steady_state_given_env!`/`solve_transition_given_env_path!` don't
need this —
their backward/forward calls see the same `(V_end, env)` by
construction.
"""
function invalidate!(buffer::AbstractStageBuffer)
    buffer.cache.kernel_valid = false
    return buffer
end
invalidate!(stage::AbstractStage) = (invalidate!(stage.buffer); stage)

"""
    allocate(spec, T = spec.element_type) -> Buffer

Build a fresh per-call buffer for `spec`. Each concrete `<X>StageSpec`
implements its own method; the generic fallback raises. `T` is the
buffer eltype; defaults to the Spec's `element_type` field
(`Float64` for most stages).

Users typically don't call `allocate` directly — `<X>Stage(...)`
calls it on construction. `allocate(spec)` is exposed for lift
authors who rebuild buffers under a different eltype (`with_eltype`).
"""
function allocate(spec::AbstractStageSpec, ::Type{T} = Float64) where {T}
    error("allocate not implemented for $(typeof(spec))")
end

# Bundled-stage delegate: build a fresh buffer at the stage's eltype.
allocate(stage::AbstractStage, ::Type{T} = Float64) where {T} =
    allocate(stage.spec, T)

"""
    backward!(spec, V_end, env, buffer) -> V_start
    backward!(stage, V_end, env)        -> V_start

Backward pass: given the value function at the *end* of the stage,
the runtime `env`, and a Buffer, populate `buffer.kernel` and
`buffer.V_start` and return the latter. Also stamps the buffer's
cache (`_seat_cache!`) so a later `forward!` can verify freshness.

The Spec-keyed signature is the primary implementation; the
bundled-stage delegate is a one-liner.
"""
function backward!(spec::AbstractStageSpec, V_end, env, buffer)
    error("backward! not implemented for $(typeof(spec))")
end

backward!(stage::AbstractStage, V_end, env) =
    backward!(stage.spec, V_end, env, stage.buffer)

"""
    forward!(spec,  Λ_start, buffer)                       -> Λ_end
    forward!(stage, Λ_start)                               -> Λ_end
    forward!(stage, Λ_start, V_end, env; kwargs...)        -> Λ_end

Forward pass: consume `Λ_start` and the kernel data populated by
the most recent `backward!`, produce `Λ_end`.

The three-arg `forward!(spec, Λ_start, buffer)` is the trusted path
— no checks. The two-arg `forward!(stage, Λ_start)` is its
bundled-stage sugar.

The four-arg `forward!(stage, Λ_start, V_end, env; kwargs...)` does
a cache check: if `buffer.cache` agrees with `(V_end, env)` it runs
trusted; on a mismatch it either errors (default,
`reseat_if_stale=false`) or re-runs `backward!` first
(`reseat_if_stale=true`). `check=false` skips the check entirely
(power-user opt-out).
"""
function forward!(spec::AbstractStageSpec, Λ_start, buffer)
    error("forward! not implemented for $(typeof(spec))")
end

forward!(stage::AbstractStage, Λ_start) =
    forward!(stage.spec, Λ_start, stage.buffer)

function forward!(stage::AbstractStage, Λ_start, V_end, env;
                  reseat_if_stale::Bool = false,
                  check::Bool           = true)
    if check
        c = stage.buffer.cache
        fresh = c.kernel_valid &&
                hash(V_end) == c.last_V_hash &&
                isequal(env, c.last_env)
        if !fresh
            if reseat_if_stale
                backward!(stage, V_end, env)   # re-seat
            else
                error("forward!: cached kernel is stale for this (V_end, env). " *
                      "Pass `reseat_if_stale=true` to recompute the kernel, " *
                      "or call `backward!` first. To skip the check entirely " *
                      "(use only when the kernel is known fresh), pass `check=false`.")
            end
        end
    end
    return forward!(stage, Λ_start)
end

"""CLAUDE
Helper for symmetric-layout stages: allocate `V_start` and `Λ_end`
buffers of the same shape and eltype. Returns the pair
destructurable as `(; Vs, Λe)`.
"""
function _alloc_VΛ(layout::StateLayout, ::Type{T}) where {T}
    dims = layout_size(layout)
    Vs   = zeros(T, dims)
    Λe   = zeros(T, dims)
    return (; Vs, Λe)
end

"""CLAUDE
Variant of `_alloc_VΛ` that accepts optional pre-allocated buffers
(used by `ProductStage` when stitching components into a fused
tensor as views). Returns `(; Vs, Λe)`. No type-equality assertion:
the product fallback for asymmetric-layout components can pass a
SubArray V_start and `nothing` for Λ_end, which becomes a fresh
Array — downstream `backward!`/`forward!` handle the heterogeneity
via dispatch.
"""
function _alloc_VΛ(layout::StateLayout, ::Type{T}, V_start, Λ_end) where {T}
    dims = layout_size(layout)
    Vs   = @something V_start zeros(T, dims)
    Λe   = @something Λ_end   zeros(T, dims)
    return (; Vs, Λe)
end

# Buffer-accessor compatibility shims #
#-------------------------------------#

"""
    V_start_buffer(stage_or_spec_with_buffer) -> AbstractArray
    Λ_end_buffer(stage_or_spec_with_buffer)   -> AbstractArray

Read the layout-shaped output buffer. The Spec layer doesn't have
buffers; the bundled Stage does (`stage.buffer.V_start`). These
helpers exist for sequence-space and lift code that pre-dates the
refactor and benefits from explicit names.
"""
V_start_buffer(stage::AbstractStage) = stage.buffer.V_start
Λ_end_buffer(stage::AbstractStage)   = stage.buffer.Λ_end

# Dependency machinery #
#----------------------#

"""
    static_env_deps(::Type{<:AbstractStageSpec}) -> NamedTuple

The `env` fields the *Spec type* itself reads, regardless of user
closures. Concrete spec types override; default is `NamedTuple()`.
"""
static_env_deps(::Type{<:AbstractStageSpec}) = NamedTuple()

"""
    effective_env_slice(spec_or_stage) -> NTuple{N, Symbol}

Names of `env` fields read by this stage. Union of
`static_env_deps(typeof(spec))` and any swept `Param` keys on the
Spec's fields. Closure-borne env reads are not introspected; they
surface as `getproperty` errors at the first backward/forward call.
"""
function effective_env_slice(spec::AbstractStageSpec)
    static = keys(static_env_deps(typeof(spec)))
    swept  = _swept_param_keys(spec)
    return Tuple(unique((static..., swept...)))
end

effective_env_slice(stage::AbstractStage) = effective_env_slice(stage.spec)

# Walk runtime Spec fields looking for swept Params. Called once at
# chain construction, not in hot paths.
function _swept_param_keys(spec::AbstractStageSpec)
    syms = Symbol[]
    for fn in fieldnames(typeof(spec))
        f = getfield(spec, fn)
        if f isa Param && is_swept(f)
            push!(syms, swept_key(f)::Symbol)
        end
    end
    return Tuple(syms)
end

"""
    validate_env(spec_or_stage, env::NamedTuple) -> Nothing

Check that `env` provides every field in
`effective_env_slice(spec_or_stage)`. Throws an informative error
listing the missing keys; returns `nothing` on success.
"""
function validate_env(spec::AbstractStageSpec, env)
    needed = effective_env_slice(spec)
    missing_keys = Symbol[]
    for k in needed
        haskey(env, k) || push!(missing_keys, k)
    end
    isempty(missing_keys) ||
        error("env is missing required fields: $(missing_keys); provided keys: $(keys(env))")
    return nothing
end

validate_env(stage::AbstractStage, env) = validate_env(stage.spec, env)

"""
    env_schema(spec_or_stage) -> NamedTuple

A prototype NamedTuple whose field names are
`effective_env_slice(spec)`. Values are `nothing`. Used by
[`make_env`](@ref) to validate user-supplied env construction.
"""
function env_schema(spec::AbstractStageSpec)
    names = effective_env_slice(spec)
    vals  = ntuple(_ -> nothing, length(names))
    return NamedTuple{names}(vals)
end
env_schema(stage::AbstractStage) = env_schema(stage.spec)

"""
    make_env(spec_or_stage; kwargs...) -> NamedTuple

Construct an env NamedTuple from kwargs, validated against
`effective_env_slice(spec)`. Errors if any required field is
missing.

The construction is permissive about extra fields (they pass
through) — closures can read whatever the user supplied — but
strict about the required minimum.
"""
function make_env(spec::AbstractStageSpec; kwargs...)
    needed = effective_env_slice(spec)
    provided = keys(kwargs)
    missing_keys = Symbol[]
    for k in needed
        k in provided || push!(missing_keys, k)
    end
    isempty(missing_keys) ||
        error("make_env: missing required env fields: $(missing_keys); " *
              "provided: $(collect(provided)); spec requires $(needed)")
    return NamedTuple(kwargs)
end
make_env(stage::AbstractStage; kwargs...) = make_env(stage.spec; kwargs...)

# `bundle(spec)` — construct the matching Stage from a Spec.
# Each concrete Spec type implements its own method. The default
# raises so misuse is caught immediately.

"""
    bundle(spec) -> stage

Construct the bundled `<X>Stage` matching `<X>StageSpec`. Each
concrete Spec type defines its own method.
"""
function bundle(spec::AbstractStageSpec)
    error("bundle not implemented for $(typeof(spec))")
end
