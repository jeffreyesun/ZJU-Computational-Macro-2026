"""
State-only flow-utility stage: `V_in[s] = u(cell; env) + V_out[s]` on
the backward pass, identity on Λ on the forward pass. The K-operator is
the identity on measures; the flow payoff enters only on the V side.
Layout is preserved (`input_layout == output_layout`); kernel and
scratch are empty. Duality is just
`⟨V_in, Λ_in⟩ = ⟨V_out, Λ_in⟩ + ⟨u, Λ_in⟩`.

`utility` is a `(cell; env)` closure (state-independent is fine — write
`(cell; env) -> log(env.c)` and ignore `cell`).
"""
struct UtilityStage{F, T<:Real, N, L<:StateLayout,
                    AV<:AbstractArray{T, N}} <: AbstractStage
    utility       :: F
    input_layout  :: L
    output_layout :: L
    V_start       :: AV
    Λ_end         :: AV
end

"Construct a [`UtilityStage`](@ref) on `layout` with the given utility closure."
function UtilityStage(layout::StateLayout;
                      utility,
                      element_type::Type{T} = Float64,
                      V_start::Union{Nothing, AbstractArray} = nothing,
                      Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    (; Vs, Λe) = _alloc_VΛ(layout, T, V_start, Λ_end)
    return UtilityStage{typeof(utility), T, ndims(Vs),
                        typeof(layout), typeof(Vs)}(
        utility, layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:UtilityStage}) = NamedTuple()

allocate(::UtilityStage, ::Type = Float64) = (; kernel = nothing, scratch = nothing)

# Backward: V_start[s] = u(cell; env) + V_end[s].
function backward!(stage::UtilityStage, V_end, env, buffers)
    (; V_start, input_layout) = stage
    for (idx, cell) in cells(input_layout)
        ci = CartesianIndex(values(idx))
        V_start[ci] = stage.utility(cell; env = env) + V_end[ci]
    end
    return V_start
end

# Forward: identity on measures.
function forward!(stage::UtilityStage, Λ_start, buffers, moments = nothing)
    copyto!(stage.Λ_end, Λ_start)
    return stage.Λ_end
end
