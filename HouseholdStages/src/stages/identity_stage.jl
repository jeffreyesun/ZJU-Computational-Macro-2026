"""
    IdentityStage{T, N, L<:StateLayout} <: AbstractStage

A no-op stage whose K is the identity operator on `M(S)`. Useful as a
component inside `product` when one branch performs no within-period
action.
"""
struct IdentityStage{T<:Real, N, L<:StateLayout,
                     AV<:AbstractArray{T,N}} <: AbstractStage
    input_layout  :: L
    output_layout :: L
    V_start       :: AV
    Λ_end         :: AV
end

"Construct an [`IdentityStage`](@ref) on `layout`."
function IdentityStage(layout::StateLayout;
                       element_type::Type{T} = Float64,
                       V_start::Union{Nothing, AbstractArray} = nothing,
                       Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    dims = layout_size(layout)
    Vs   = V_start === nothing ? zeros(T, dims) : V_start
    Λe   = Λ_end   === nothing ? zeros(T, dims) : Λ_end
    @assert typeof(Vs) === typeof(Λe) "IdentityStage: V_start and Λ_end must have the same concrete array type"
    return IdentityStage{T, length(dims), typeof(layout), typeof(Vs)}(
        layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:IdentityStage}) = NamedTuple()

allocate(::IdentityStage{T,N}, ::Type{T2} = T) where {T,N,T2} = (nothing, nothing)

function backward!(stage::IdentityStage{T,N},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {T,N}
    copyto!(stage.V_start, V_end)
    return stage.V_start
end

function forward!(stage::IdentityStage{T,N},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {T,N}
    copyto!(stage.Λ_end, Λ_start)
    return stage.Λ_end
end
