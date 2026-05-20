"""
Configuration for a [`UtilityStage`](@ref): state-only flow-utility
stage with `V_in[s] = u(cell; env) + V_out[s]` on backward and identity
on Λ on forward. The K-operator is the identity on measures; the flow
payoff enters only on the V side. Pure data — no per-call buffers.

`utility` is a `(cell; env)` closure (state-independent is fine — write
`(cell; env) -> log(env.c)` and ignore `cell`).
"""
struct UtilityStageSpec{F, T<:Real, L<:StateLayout} <: AbstractStageSpec
    utility       :: F
    input_layout  :: L
    output_layout :: L
    element_type  :: Type{T}
end

"""
    UtilityStageSpec(layout; utility, element_type=Float64)

Build the Spec for a utility stage on `layout` with the given utility
closure.
"""
function UtilityStageSpec(layout::StateLayout;
                          utility,
                          element_type::Type{T} = Float64) where {T<:Real}
    return UtilityStageSpec{typeof(utility), T, typeof(layout)}(
        utility, layout, layout, element_type,
    )
end

"""
Per-call buffer for a utility stage. Kernel = `nothing` (K is the
identity on measures). Scratch = `nothing` (no workspace needed). Holds
`V_start` and `Λ_end` shaped by the (common) layout.
"""
struct UtilityStageBuffer{T<:Real, N, AV<:AbstractArray{T,N}} <: AbstractStageBuffer
    kernel  :: Nothing
    scratch :: Nothing
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
State-only flow-utility stage. Construct via
`UtilityStage(layout; utility = (cell; env) -> ...)`. Layout is
preserved; the K-operator is the identity on measures and the flow
payoff enters only on the V side.
"""
struct UtilityStage{Spec<:UtilityStageSpec, Buffer<:UtilityStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function UtilityStage(layout::StateLayout;
                      utility,
                      element_type::Type{T} = Float64,
                      V_start::Union{Nothing, AbstractArray} = nothing,
                      Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    spec = UtilityStageSpec(layout; utility, element_type)
    return UtilityStage(spec, allocate(spec, T; V_start, Λ_end))
end

UtilityStage(spec::UtilityStageSpec) = UtilityStage(spec, allocate(spec))
bundle(spec::UtilityStageSpec)       = UtilityStage(spec)

static_env_deps(::Type{<:UtilityStageSpec}) = NamedTuple()

# Allocate #
#----------#

function allocate(spec::UtilityStageSpec, ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    return UtilityStageBuffer{T, ndims(Vs), typeof(Vs)}(
        nothing, nothing, Vs, Λe, CacheState(),
    )
end

# Backward: V_start[s] = u(cell; env) + V_end[s].
function backward!(spec::UtilityStageSpec, V_end, env, buffer::UtilityStageBuffer)
    V_start = buffer.V_start
    for (idx, cell) in cells(spec.input_layout)
        ci = CartesianIndex(values(idx))
        V_start[ci] = spec.utility(cell; env = env) + V_end[ci]
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward: identity on measures.
function forward!(spec::UtilityStageSpec, Λ_start, buffer::UtilityStageBuffer)
    copyto!(buffer.Λ_end, Λ_start)
    return buffer.Λ_end
end
