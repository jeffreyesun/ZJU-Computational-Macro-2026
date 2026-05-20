"""
Configuration for a hard discrete-choice stage. Actions are the levels
of `choice_axis`. The K-operator is a sparse permutation (`δ_{π(s)}`
for each input cell, where `π(s)` is the chosen-action's next-cell
index); the materialised integer policy lives on the Buffer.

`flow_payoff(cell, action; env)` is the period payoff at the cell of
taking action `action`; `-Inf` is treated as "unavailable" and skipped.
`next_state_idx(cell, action) -> Int` returns the integer index along
`choice_axis` of the cell reached by `action`.

Pure data — no per-call buffers.
"""
struct ArgmaxStageSpec{F, BF, T<:Real,
                       LIn<:StateLayout, LOut<:StateLayout} <: AbstractStageSpec
    choice_axis    :: Symbol
    choice_dim     :: Int
    flow_payoff    :: F
    next_state_idx :: BF
    input_layout   :: LIn
    output_layout  :: LOut
    element_type   :: Type{T}
end

"""
    ArgmaxStageSpec(layout; choice_axis, flow_payoff, next_state_idx,
                    element_type=Float64)

Build the Spec for an [`ArgmaxStage`](@ref).
"""
function ArgmaxStageSpec(layout::StateLayout;
                         choice_axis::Symbol,
                         flow_payoff,
                         next_state_idx,
                         element_type::Type{T} = Float64) where {T<:Real}
    choice_dim = axis_position(layout, choice_axis)
    return ArgmaxStageSpec{typeof(flow_payoff), typeof(next_state_idx),
                           T, typeof(layout), typeof(layout)}(
        choice_axis, choice_dim, flow_payoff, next_state_idx,
        layout, layout, element_type,
    )
end

"""
Per-call buffer for a hard discrete-choice stage. The kernel is a
NamedTuple `(; policy)` holding the materialised integer policy array
(one chosen action index per cell). No scratch.
"""
struct ArgmaxStageBuffer{T<:Real, N, AV<:AbstractArray{T,N},
                         Kernel} <: AbstractStageBuffer
    kernel  :: Kernel
    scratch :: Nothing
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A hard discrete-choice stage. Construct via
`ArgmaxStage(layout; choice_axis, flow_payoff, next_state_idx)`.
Composes via `∘` and `×`.
"""
struct ArgmaxStage{Spec<:ArgmaxStageSpec,
                   Buffer<:ArgmaxStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function ArgmaxStage(layout::StateLayout;
                     choice_axis::Symbol,
                     flow_payoff,
                     next_state_idx,
                     element_type::Type{T} = Float64,
                     V_start::Union{Nothing, AbstractArray} = nothing,
                     Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    spec = ArgmaxStageSpec(layout; choice_axis, flow_payoff, next_state_idx, element_type)
    return ArgmaxStage(spec, allocate(spec, T; V_start, Λ_end))
end

ArgmaxStage(spec::ArgmaxStageSpec) = ArgmaxStage(spec, allocate(spec))
bundle(spec::ArgmaxStageSpec)      = ArgmaxStage(spec)

static_env_deps(::Type{<:ArgmaxStageSpec}) = NamedTuple()

# Allocate #
#----------#

function allocate(spec::ArgmaxStageSpec, ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    policy = zeros(Int, size(Vs))
    kernel = (; policy)
    return ArgmaxStageBuffer{T, ndims(Vs), typeof(Vs), typeof(kernel)}(
        kernel, nothing, Vs, Λe, CacheState(),
    )
end

# Backward #
#----------#

function backward!(spec::ArgmaxStageSpec, V_end, env, buffer::ArgmaxStageBuffer)
    (; input_layout, choice_dim) = spec
    V_start = buffer.V_start
    policy  = buffer.kernel.policy
    actions = axisvalues(input_layout.axes[choice_dim])
    T       = eltype(V_start)

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)

        best_v   = typemin(T)
        best_a_i = 0
        for (a_i, action) in pairs(actions)
            r = spec.flow_payoff(cell, action; env = env)
            isfinite(r) || continue
            next_axis_i = spec.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            v = r + V_end[CartesianIndex(out_idxs)]
            if v > best_v
                best_v   = v
                best_a_i = a_i
            end
        end
        best_a_i == 0 && error("ArgmaxStage: no finite-payoff action at cell $cell")

        V_start[ci_in] = best_v
        policy[ci_in]  = best_a_i
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward #
#---------#

function forward!(spec::ArgmaxStageSpec, Λ_start, buffer::ArgmaxStageBuffer)
    (; input_layout, choice_dim) = spec
    Λ_end  = buffer.Λ_end
    policy = buffer.kernel.policy
    actions = axisvalues(input_layout.axes[choice_dim])
    T = eltype(Λ_end)

    fill!(Λ_end, zero(T))
    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        mass    = Λ_start[ci_in]
        iszero(mass) && continue

        chosen_action = actions[policy[ci_in]]
        next_axis_i   = spec.next_state_idx(cell, chosen_action)
        out_idxs      = Base.setindex(in_idxs, next_axis_i, choice_dim)
        Λ_end[CartesianIndex(out_idxs)] += mass
    end
    return Λ_end
end
