###################
# Stage protocol #
###################
#
# Three layers. A `Spec` is pure stage-specific configuration
# (immutable, no buffers, no layout). A `Buffer` is per-call state —
# the materialised K-operator data, internal scratch, the V/Λ
# output arrays, the layouts the buffer was allocated against, and
# a freshness cache. A `Stage` bundles one of each and is the
# user-facing object.
#
# Specs are layout-free under the 2026-05-25 refactor: layouts
# flow into `allocate(spec, layout, T)` and end up on the buffer.
# A user-authored stage defines its Spec, optionally a `Kernel`
# and/or `Scratch` struct, `allocate_kernel`, `backward!`,
# `forward!`, and emits a wrapper struct via `@definestage`.

"Pure stage configuration. Layout-free; one struct per stage class."
abstract type AbstractStageSpec end

"Per-call buffer: kernel + scratch + V/Λ outputs + layouts + cache."
abstract type AbstractStageBuffer end

"Bundle of one Spec and one Buffer; the user-facing layer."
abstract type AbstractStage end

# Cache fingerprint #
#-------------------#

"""
Mutable record of the `(V_end, env)` last seen by `backward!`. A
later `forward!` can check whether its kernel is still valid for
the `(V_end, env)` it's being given.
"""
mutable struct CacheState
    last_V_hash  :: UInt
    last_env     :: Any
    kernel_valid :: Bool
end
CacheState() = CacheState(zero(UInt), nothing, false)

"""
Stamp the buffer's cache with the `(V_end, env)` `backward!` just
consumed. Call at the end of every concrete `backward!`.
"""
function _seat_cache!(buffer::AbstractStageBuffer, V_end, env)
    c = buffer.cache
    c.last_V_hash  = hash(V_end)
    c.last_env     = env
    c.kernel_valid = true
    return buffer
end

"Mark the cache stale; the next cache-checking `forward!` will refuse to run trusted."
function invalidate!(buffer::AbstractStageBuffer)
    buffer.cache.kernel_valid = false
    return buffer
end
invalidate!(stage::AbstractStage) = (invalidate!(stage.buffer); stage)

# Generic buffer #
#----------------#

"""
The library-provided buffer shape. Per-stage Buffer structs from
the pre-2026-05-25 era are replaced by this; `Kernel` and `Scratch`
are free type parameters so each stage chooses its own.
"""
struct StageBuffer{Kernel, Scratch, V<:AbstractArray, Λ<:AbstractArray,
                   LIn, LOut} <: AbstractStageBuffer
    kernel        :: Kernel
    scratch       :: Scratch
    V_start       :: V
    Λ_end         :: Λ
    input_layout  :: LIn
    output_layout :: LOut
    cache         :: CacheState
end

# Allocation protocol #
#---------------------#

"User overrides this to allocate the kernel cache. Default: `nothing`."
allocate_kernel(::AbstractStageSpec, ::Type, ::StateLayout) = nothing

"User overrides this to allocate compute scratch. Default: `nothing`."
allocate_scratch(::AbstractStageSpec, ::Type, ::StateLayout) = nothing

"Input layout the spec sees. Default: the layout passed in. Override for stages whose input layout differs structurally."
input_layout(::AbstractStageSpec, layout::StateLayout) = layout

"Output layout the spec produces. Default: same as input. `ForgetfulSumStage` overrides to drop the named axis."
output_layout(::AbstractStageSpec, layout::StateLayout) = layout

"Default buffer eltype. Override for specs that infer T from a parameter or array field."
default_eltype(::AbstractStageSpec) = Float64

"""
Allocate a fresh `StageBuffer` for `spec` against `layout`. The
optional `V_start` / `Λ_end` kwargs are library-internal — used by
`ProductStage` for view-stitching into a fused tensor; user-facing
constructors never expose them.
"""
function allocate(spec::AbstractStageSpec, layout::StateLayout,
                  ::Type{T} = default_eltype(spec);
                  V_start=nothing, Λ_end=nothing) where {T}
    L_in  = input_layout(spec, layout)
    L_out = output_layout(spec, layout)
    Vs = @something V_start zeros(T, layout_size(L_in))
    Λe = @something Λ_end   zeros(T, layout_size(L_out))
    kernel  = allocate_kernel(spec, T, layout)
    scratch = allocate_scratch(spec, T, layout)
    return StageBuffer(kernel, scratch, Vs, Λe, L_in, L_out, CacheState())
end

# Stage-keyed delegates — buffer-first dispatch on the spec-keyed methods #
#------------------------------------------------------------------------#

"`backward!(stage, V_end, env) -> V_start`. Bundled-stage delegate."
backward!(stage::AbstractStage, V_end, env) =
    backward!(stage.buffer, stage.spec, V_end, env)

"`forward!(stage, Λ_start) -> Λ_end`. Bundled-stage delegate."
forward!(stage::AbstractStage, Λ_start) =
    forward!(stage.buffer, stage.spec, Λ_start)

"""
    forward!(stage, Λ_start, V_end, env; check=true, reseat_if_stale=false) -> Λ_end

Cache-checking variant: refuses to run trusted unless the buffer's
cache agrees with `(V_end, env)`. With `reseat_if_stale=true`,
re-runs `backward!` to refresh; otherwise errors. `check=false`
skips the check (power-user opt-out).
"""
function forward!(stage::AbstractStage, Λ_start, V_end, env;
                  reseat_if_stale::Bool=false, check::Bool=true)
    if check
        c = stage.buffer.cache
        fresh = c.kernel_valid && hash(V_end) == c.last_V_hash &&
                isequal(env, c.last_env)
        if !fresh
            reseat_if_stale || error("forward!: cached kernel is stale for this (V_end, env). " *
                                     "Pass reseat_if_stale=true or call backward! first.")
            backward!(stage, V_end, env)
        end
    end
    return forward!(stage, Λ_start)
end

# Public accessors #
#------------------#

"`input_layout(stage)` — the layout the stage's buffer was allocated against."
input_layout(stage::AbstractStage)  = stage.buffer.input_layout
output_layout(stage::AbstractStage) = stage.buffer.output_layout

"Read the layout-shaped output buffers."
V_start_buffer(stage::AbstractStage) = stage.buffer.V_start
Λ_end_buffer(stage::AbstractStage)   = stage.buffer.Λ_end

# `@definestage` — wrapper struct + constructors + bundle #
#---------------------------------------------------------#

"""
Stamp out the per-stage wrapper struct, outer constructors, and
`bundle` method. Usage:

```julia
@definestage ArgmaxStage ArgmaxStageSpec kernel=ArgmaxKernel
@definestage MarkovStage MarkovStageSpec scratch=MarkovScratch
@definestage IdentityStage IdentityStageSpec
```

`kernel=K` tightens the Buffer constraint to `StageBuffer{<:K}`;
`scratch=S` tightens the Scratch parameter. Both default to
unconstrained.
"""
macro definestage(stage_name, spec_name, opts...)
    kernel_type = :Any
    scratch_type = :Any
    for opt in opts
        @assert opt isa Expr && opt.head === :(=) "@definestage: expected key=Value, got $opt"
        key, val = opt.args
        if key === :kernel
            kernel_type = val
        elseif key === :scratch
            scratch_type = val
        else
            error("@definestage: unknown option $key")
        end
    end
    buf_constraint = :(StageBuffer{<:$(esc(kernel_type)), <:$(esc(scratch_type))})
    spec_e = esc(spec_name)
    stg_e  = esc(stage_name)
    bundle_e = esc(:bundle)
    return quote
        struct $stg_e{Spec<:$spec_e, Buffer<:$buf_constraint} <: AbstractStage
            spec   :: Spec
            buffer :: Buffer
        end
        $stg_e(spec::$spec_e, layout::StateLayout, ::Type{T}=default_eltype(spec)) where {T} =
            $stg_e(spec, allocate(spec, layout, T))
        $stg_e(layout::StateLayout; kwargs...) =
            $stg_e($spec_e(; kwargs...), layout)
        $bundle_e(spec::$spec_e, layout::StateLayout) = $stg_e(spec, layout)
        $bundle_e(spec::$spec_e, layout::StateLayout, ::Type{T}) where {T} =
            $stg_e(spec, layout, T)
    end
end

# `bundle(spec, layout)` — generic fallback raises until @definestage emits the method.
function bundle(spec::AbstractStageSpec, ::StateLayout)
    error("bundle not implemented for $(typeof(spec))")
end

# Dependency machinery #
#----------------------#

"The `env` fields the Spec type itself reads. Concrete specs override."
static_env_deps(::Type{<:AbstractStageSpec}) = NamedTuple()

"Names of `env` fields read by this stage — static deps ∪ env-resolved spec fields."
function effective_env_slice(spec::AbstractStageSpec)
    static = keys(static_env_deps(typeof(spec)))
    return Tuple(unique((static..., _env_field_names(spec)...)))
end
effective_env_slice(stage::AbstractStage) = effective_env_slice(stage.spec)

"""
Marker wrapping a Symbol that names an `env` field. Spec fields hold
either a literal value or a `FromEnv(:key)`; `resolve` dispatches on
the value's type.
"""
struct FromEnv
    key :: Symbol
end

"Resolve a stage-parameter field: pass through if literal, look up in `env` if `FromEnv`."
resolve(val, env)            = val
resolve(fe::FromEnv, env)    = env[fe.key]
# Pass-through when env is absent — lets `resolve(buffer, spec)` work
# even with FromEnv fields the caller won't destructure.
resolve(fe::FromEnv, ::Nothing) = fe

"Names of `env` fields the spec currently reads — the `FromEnv` markers held in any field."
function _env_field_names(spec::AbstractStageSpec)
    syms = Symbol[]
    for fn in fieldnames(typeof(spec))
        v = getfield(spec, fn)
        v isa FromEnv && push!(syms, v.key)
    end
    return Tuple(syms)
end

"""
Project a spec's fields (env-resolved via `resolve`) and the buffer's
layout/output geometry into a single NamedTuple. The destructure target
for `backward!` / `forward!` heads. Geometry wins on any name clash.
"""
function resolve(buffer::AbstractStageBuffer, spec::AbstractStageSpec, env=nothing)
    spec_nt = NamedTuple{fieldnames(typeof(spec))}(
        ntuple(i -> resolve(getfield(spec, i), env), nfields(spec))
    )
    layout = buffer.input_layout
    geom = (input_layout  = layout,
            output_layout = buffer.output_layout,
            dims          = layout_size(layout),
            V_start       = buffer.V_start,
            Λ_end         = buffer.Λ_end)
    return merge(spec_nt, geom)
end

"Check that `env` provides every field in `effective_env_slice(spec)`."
function validate_env(spec::AbstractStageSpec, env)
    needed = effective_env_slice(spec)
    missing_keys = Symbol[k for k in needed if !haskey(env, k)]
    isempty(missing_keys) ||
        error("env is missing required fields: $missing_keys; provided: $(keys(env))")
    return nothing
end
validate_env(stage::AbstractStage, env) = validate_env(stage.spec, env)

"Prototype NamedTuple whose names are `effective_env_slice(spec)`; values `nothing`."
function env_schema(spec::AbstractStageSpec)
    names = effective_env_slice(spec)
    return NamedTuple{names}(ntuple(_ -> nothing, length(names)))
end
env_schema(stage::AbstractStage) = env_schema(stage.spec)

"Construct an env NamedTuple from kwargs, validated against `effective_env_slice(spec)`."
function make_env(spec::AbstractStageSpec; kwargs...)
    needed   = effective_env_slice(spec)
    provided = keys(kwargs)
    missing_keys = Symbol[k for k in needed if !(k in provided)]
    isempty(missing_keys) ||
        error("make_env: missing required env fields: $missing_keys; provided: $(collect(provided))")
    return NamedTuple(kwargs)
end
make_env(stage::AbstractStage; kwargs...) = make_env(stage.spec; kwargs...)

# Numerical helpers #
#-------------------#

"""
In-place numerically-stable softmax of `U / ε` along its trailing
axis. Overwrites `U` with the choice probabilities and writes the
per-row `ε · log Σ exp(U[i,:]/ε)` into `lse` (shaped like `U`'s
leading axes). At least one entry per row must be finite.
"""
function _softmax_and_lse_along_last!(lse::AbstractArray{T},
                                      U::AbstractArray, ε::Real) where {T}
    Uflat   = reshape(U, :, last(size(U)))
    lseflat = vec(lse)
    for (s, u_row) in enumerate(eachrow(Uflat))
        m = maximum(u_row)
        @assert isfinite(m)
        @. u_row = exp((u_row - m) / ε)
        denom = sum(u_row)
        u_row ./= denom
        lseflat[s] = m + ε * log(denom)
    end
    return lse
end
