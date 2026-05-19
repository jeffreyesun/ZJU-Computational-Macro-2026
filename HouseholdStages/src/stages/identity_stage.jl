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
    (; Vs, Λe) = _alloc_VΛ(layout, T, V_start, Λ_end)
    return IdentityStage{T, ndims(Vs), typeof(layout), typeof(Vs)}(
        layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:IdentityStage}) = NamedTuple()

allocate(::IdentityStage, ::Type = Float64) = (; kernel = nothing, scratch = nothing)

function backward!(stage::IdentityStage, V_end, env, buffers)
    copyto!(stage.V_start, V_end)
    return stage.V_start
end

function forward!(stage::IdentityStage, Λ_start, buffers, moments = nothing)
    copyto!(stage.Λ_end, Λ_start)
    return stage.Λ_end
end
