"""
Configuration for a logit-smoothed discrete-choice stage. Actions are
the levels of `choice_axis`. The K-operator is the stochastic kernel
`Σ_a π(a|s) δ_{σ(s,a)}` with `π(a|s) = softmax((u_a) / ε)`; the
materialised probability tensor lives on the Buffer.

`flow_payoff(cell, action; env)` and `next_state_idx(cell, action) -> Int`
mirror the [`ArgmaxStage`](@ref) convention. `ε::Param{T}` is the logit
scale, either a literal or a Symbol-valued sweep key (see [`Param`](@ref)).
`-Inf` flow payoffs are treated as "unavailable action": skipped in the
log-sum-exp and assigned zero probability.

Pure data — no per-call buffers.
"""
struct LogitChoiceStageSpec{F, BF, T<:Real,
                            LIn<:StateLayout, LOut<:StateLayout} <: AbstractStageSpec
    choice_axis    :: Symbol
    choice_dim     :: Int
    flow_payoff    :: F
    next_state_idx :: BF
    ε              :: Param{T}
    input_layout   :: LIn
    output_layout  :: LOut
    element_type   :: Type{T}
end

"""
    LogitChoiceStageSpec(layout; choice_axis, flow_payoff, next_state_idx,
                         ε=Param(1.0), element_type=nothing)

Build the Spec for a [`LogitChoiceStage`](@ref). `element_type`
defaults to `eltype(ε)`.
"""
function LogitChoiceStageSpec(layout::StateLayout;
                              choice_axis::Symbol,
                              flow_payoff,
                              next_state_idx,
                              ε::Param{T} = Param(1.0),
                              element_type::Union{Type, Nothing} = nothing) where {T<:Real}
    Tb = @something element_type T
    Tb === T ||
        error("LogitChoiceStageSpec: element_type ($Tb) must match eltype(ε) ($T); " *
              "use `with_eltype` to switch eltypes instead.")
    choice_dim = axis_position(layout, choice_axis)
    return LogitChoiceStageSpec{typeof(flow_payoff), typeof(next_state_idx),
                                Tb, typeof(layout), typeof(layout)}(
        choice_axis, choice_dim, flow_payoff, next_state_idx, ε,
        layout, layout, Tb,
    )
end

"""
Per-call buffer for a logit-smoothed discrete-choice stage. The kernel
is a NamedTuple `(; choice_prob)` holding the per-cell action
probability tensor of shape `(layout_size..., n_actions)`. No scratch.
"""
struct LogitChoiceStageBuffer{T<:Real, N, AV<:AbstractArray{T,N},
                              Kernel} <: AbstractStageBuffer
    kernel  :: Kernel
    scratch :: Nothing
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A logit-smoothed discrete-choice stage. Construct via
`LogitChoiceStage(layout; choice_axis, flow_payoff, next_state_idx, ε)`.
Composes via `∘` and `×`.
"""
struct LogitChoiceStage{Spec<:LogitChoiceStageSpec,
                        Buffer<:LogitChoiceStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function LogitChoiceStage(layout::StateLayout;
                          choice_axis::Symbol,
                          flow_payoff,
                          next_state_idx,
                          ε::Param{T} = Param(1.0),
                          element_type::Union{Type, Nothing} = nothing,
                          V_start::Union{Nothing, AbstractArray} = nothing,
                          Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    spec = LogitChoiceStageSpec(layout; choice_axis, flow_payoff, next_state_idx,
                                ε, element_type)
    Tb = spec.element_type
    return LogitChoiceStage(spec, allocate(spec, Tb; V_start, Λ_end))
end

LogitChoiceStage(spec::LogitChoiceStageSpec) = LogitChoiceStage(spec, allocate(spec))
bundle(spec::LogitChoiceStageSpec)           = LogitChoiceStage(spec)

static_env_deps(::Type{<:LogitChoiceStageSpec}) = NamedTuple()

# Allocate #
#----------#

function allocate(spec::LogitChoiceStageSpec, ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    dims      = layout_size(spec.input_layout)
    n_actions = axissize(spec.input_layout.axes[spec.choice_dim])
    choice_prob = zeros(T, dims..., n_actions)
    kernel = (; choice_prob)
    return LogitChoiceStageBuffer{T, ndims(Vs), typeof(Vs), typeof(kernel)}(
        kernel, nothing, Vs, Λe, CacheState(),
    )
end

# Backward #
#----------#

function backward!(spec::LogitChoiceStageSpec, V_end, env,
                   buffer::LogitChoiceStageBuffer)
    (; input_layout, choice_dim) = spec
    V_start = buffer.V_start
    prob    = buffer.kernel.choice_prob
    ε       = resolve(spec.ε, env)
    actions = axisvalues(input_layout.axes[choice_dim])
    n_a     = length(actions)
    T       = eltype(V_start)

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)

        max_u   = typemin(T)
        any_fin = false
        # Pass 1: max_u for numerical stability.
        for action in actions
            r = spec.flow_payoff(cell, action; env = env)
            if isfinite(r)
                any_fin = true
                next_axis_i = spec.next_state_idx(cell, action)
                out_idxs = Base.setindex(in_idxs, next_axis_i, choice_dim)
                u = r + V_end[CartesianIndex(out_idxs)]
                if u > max_u
                    max_u = u
                end
            end
        end
        any_fin || error("LogitChoiceStage: no finite-payoff action at cell $cell")

        # Pass 2: weights / unnormalised probs.
        denom = zero(T)
        for (a_i, action) in pairs(actions)
            r = spec.flow_payoff(cell, action; env = env)
            if isfinite(r)
                next_axis_i = spec.next_state_idx(cell, action)
                out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
                u           = r + V_end[CartesianIndex(out_idxs)]
                w           = exp((u - max_u) / ε)
                prob[in_idxs..., a_i] = w
                denom += w
            else
                prob[in_idxs..., a_i] = zero(T)
            end
        end
        # Pass 3: normalise.
        for a_i in 1:n_a
            prob[in_idxs..., a_i] /= denom
        end
        V_start[ci_in] = max_u + ε * log(denom)
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward #
#---------#

function forward!(spec::LogitChoiceStageSpec, Λ_start,
                  buffer::LogitChoiceStageBuffer)
    (; input_layout, choice_dim) = spec
    Λ_end = buffer.Λ_end
    prob  = buffer.kernel.choice_prob
    actions = axisvalues(input_layout.axes[choice_dim])
    T = eltype(Λ_end)

    fill!(Λ_end, zero(T))
    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        mass    = Λ_start[ci_in]
        iszero(mass) && continue

        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = spec.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            Λ_end[CartesianIndex(out_idxs)] += mass * p
        end
    end
    return Λ_end
end
