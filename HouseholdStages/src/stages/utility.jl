"""
State-only flow-utility stage: `V_in[s] = u(cell; env) + V_out[s]` on
the backward pass, identity on Λ on the forward pass. The K-operator is
the identity on measures; the flow payoff enters only on the V side.
Layout is preserved (`input_layout == output_layout`); kernel and
scratch are empty. Duality is just
`⟨V_in, Λ_in⟩ = ⟨V_out, Λ_in⟩ + ⟨u, Λ_in⟩`.

`utility` is a `(cell; env)` closure (state-independent is fine — write
`(cell; env) -> log(env.c)` and ignore `cell`). `closure_deps` lists
the env fields it reads.
"""
struct UtilityStage{F, T<:Real, N, L<:StateLayout, D,
                    AV<:AbstractArray{T, N}} <: AbstractStage
    utility       :: F
    closure_deps  :: NTuple{D, Symbol}
    input_layout  :: L
    output_layout :: L
    V_start       :: AV
    Λ_end         :: AV
end

"Construct a [`UtilityStage`](@ref) on `layout` with the given utility closure."
function UtilityStage(layout::StateLayout;
                      utility,
                      closure_deps::NTuple{D, Symbol} = (),
                      element_type::Type{T} = Float64,
                      V_start::Union{Nothing, AbstractArray} = nothing,
                      Λ_end::Union{Nothing, AbstractArray}  = nothing) where {D, T<:Real}
    dims = layout_size(layout)
    Vs   = @something V_start zeros(T, dims)
    Λe   = @something Λ_end   zeros(T, dims)
    @assert typeof(Vs) === typeof(Λe) "UtilityStage: V_start and Λ_end must have the same concrete array type"
    return UtilityStage{typeof(utility), T, length(dims),
                        typeof(layout), D, typeof(Vs)}(
        utility, closure_deps, layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:UtilityStage}) = NamedTuple()

allocate(::UtilityStage{F,T,N}, ::Type{T2} = T) where {F,T,N,T2} =
    (nothing, nothing)

# Backward: V_start[s] = u(cell; env) + V_end[s].
function backward!(stage::UtilityStage{F,T,N},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {F,T,N}
    V_start = stage.V_start
    layout  = stage.input_layout
    for (idx, cell) in cells(layout)
        ci = CartesianIndex(values(idx))
        V_start[ci] = stage.utility(cell; env = env) + V_end[ci]
    end
    return V_start
end

# Forward: identity on measures.
function forward!(stage::UtilityStage{F,T,N},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {F,T,N}
    copyto!(stage.Λ_end, Λ_start)
    return stage.Λ_end
end
