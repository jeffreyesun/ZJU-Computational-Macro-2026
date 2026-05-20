"""
Configuration for an [`IdentityStage`](@ref): a no-op stage whose K is
the identity operator on `M(S)`. Useful as a component inside `product`
when one branch performs no within-period action. Pure data — no
per-call buffers.
"""
struct IdentityStageSpec{T<:Real, L<:StateLayout} <: AbstractStageSpec
    input_layout  :: L
    output_layout :: L
    element_type  :: Type{T}
end

"""
    IdentityStageSpec(layout; element_type=Float64)

Build the Spec for an identity stage on `layout`.
"""
function IdentityStageSpec(layout::StateLayout;
                           element_type::Type{T} = Float64) where {T<:Real}
    return IdentityStageSpec{T, typeof(layout)}(layout, layout, element_type)
end

"""
Per-call buffer for an identity stage. Kernel = `nothing` (K is the
identity, V/θ-independent and trivial). Scratch = `nothing` (no
workspace needed). Holds `V_start` and `Λ_end` shaped by the (common)
layout.
"""
struct IdentityStageBuffer{T<:Real, N, AV<:AbstractArray{T,N}} <: AbstractStageBuffer
    kernel  :: Nothing
    scratch :: Nothing
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A no-op stage whose K is the identity operator on `M(S)`. Construct via
`IdentityStage(layout)`. Useful as a component inside `product` when one
branch performs no within-period action.
"""
struct IdentityStage{Spec<:IdentityStageSpec, Buffer<:IdentityStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function IdentityStage(layout::StateLayout;
                       element_type::Type{T} = Float64,
                       V_start::Union{Nothing, AbstractArray} = nothing,
                       Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    spec = IdentityStageSpec(layout; element_type)
    return IdentityStage(spec, allocate(spec, T; V_start, Λ_end))
end

IdentityStage(spec::IdentityStageSpec) = IdentityStage(spec, allocate(spec))
bundle(spec::IdentityStageSpec)        = IdentityStage(spec)

static_env_deps(::Type{<:IdentityStageSpec}) = NamedTuple()

# Allocate #
#----------#

function allocate(spec::IdentityStageSpec, ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    return IdentityStageBuffer{T, ndims(Vs), typeof(Vs)}(
        nothing, nothing, Vs, Λe, CacheState(),
    )
end

# Backward #
#----------#

function backward!(spec::IdentityStageSpec, V_end, env, buffer::IdentityStageBuffer)
    copyto!(buffer.V_start, V_end)
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

# Forward #
#---------#

function forward!(spec::IdentityStageSpec, Λ_start, buffer::IdentityStageBuffer)
    copyto!(buffer.Λ_end, Λ_start)
    return buffer.Λ_end
end
