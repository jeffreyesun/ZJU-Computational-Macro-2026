"""
    AbstractStage

A stage is one segment of the within-period problem. Operationally, a
stage σ is a function `(V_end, env) → (r, K)` where r is a flow payoff
on the input state space and K is a Markov kernel from the input state
space to the output state space. The pair (r, K) defines the stage's
backward action on continuation values, `V_in = r + K^T V_end`, and
its forward action on distributions, `Λ_out = K Λ_in`. These two
actions are mutually adjoint under the natural pairing
`⟨V, Λ⟩ = ∫ V dΛ`. For optimizing agents, the envelope theorem yields
`∂V_in/∂V_end = K^T` at the evaluation point, so the same K computed
during backward suffices for the forward pass and for adjoint
propagation.

Households flow through a chain of stages: each stage's backward pass
takes the value function at its end, computes the stage's runtime
kernel data, and produces the value function at its start. Each
stage's forward pass takes the distribution at its start, consumes the
runtime kernel from the prior backward pass, and produces the
distribution at its end. **The forward pass does not take env** — env
was fully consumed by backward in producing the kernel.

The supertype is a lightweight tag — it carries no behavior. Concrete
stages are parametric on the concrete types of their array- and
function-typed fields (the JMP pattern); CPU vs. GPU dispatch flows
through those parametric types rather than a platform sigil.

A concrete stage struct is expected to implement:

  - [`allocate`](@ref) — produce a `buffers` workspace whose lifetime
    spans one backward → forward sweep.
  - [`backward!`](@ref) — write `V_start` and populate `buffers.kernel`
    from `V_end` and `env`.
  - [`forward!`](@ref) — write `Λ_end` from `Λ_start` and `buffers`.
  - [`V_start_buffer`](@ref) and [`Λ_end_buffer`](@ref) — accessors
    for the stage's preallocated layout-shaped output buffers.

See also: [`MarkovStage`](@ref), [`ArgmaxStage`](@ref), [`LogitChoiceStage`](@ref),
[`ChainStage`](@ref).
"""
abstract type AbstractStage end

"""
    allocate(stage, T = Float64) -> buffers

Produce the workspace bundle for one backward → forward sweep of `stage`
with element type `T`. The shape of `buffers` depends on the stage:

  * Non-chain stages return a NamedTuple `(; kernel, scratch)` where
    `kernel` holds the runtime data parameterizing the K-operator at
    the current `(V_end, env)` (e.g., integer policy index for
    `ArgmaxStage`, probability tensor for `LogitChoiceStage`) and
    `scratch` is any internal workspace that carries no morphism
    content. Either field may be `nothing` / `(;)` when unused.
  * `ChainStage` returns a `Tuple` of per-stage buffers — element `i`
    is the bundle for `c.stages[i]`.

The kernel/scratch split is preserved at the per-stage level so adjoint
and GPU code can target the morphism payload directly; in the common
user-facing case the caller never has to unpack the bundle.

Concrete stages must implement this method. The fallback raises.
"""
function allocate(stage::AbstractStage, ::Type{T} = Float64) where {T}
    error("allocate not implemented for $(typeof(stage))")
end

"""
    backward!(stage, V_end, env, buffers) -> V_start

Backward pass: given the value function at the *end* of the stage, the
runtime `env` slice, and the workspace `buffers` from `allocate(stage)`,
return the value function at the *start*.

The stage writes into the preallocated `V_start_buffer(stage)` and
populates `buffers.kernel` (or, for chains, the per-stage kernels) with
the runtime data parameterizing the K-operator — whatever the forward
pass will need to apply K. Returns the V_start buffer.

Concrete stages must implement this method. The fallback raises.
"""
function backward!(stage::AbstractStage, V_end, env, buffers)
    error("backward! not implemented for $(typeof(stage))")
end

"""
    forward!(stage, Λ_start, buffers, moments=nothing) -> Λ_end

Forward pass: given the distribution at the *start*, the workspace
`buffers` from a prior `backward!`, and an optional moments accumulator
(`nothing` if no moments are being emitted), return the distribution at
the *end*.

**Forward does not take `env`.** All env-dependent computation is
baked into `buffers.kernel` by the prior `backward!` call. The forward
pass reads the kernel and applies K to `Λ_start`; it never reads env
directly.

Concrete stages must implement this method. The fallback raises.
"""
function forward!(stage::AbstractStage, Λ_start, buffers,
                  moments = nothing)
    error("forward! not implemented for $(typeof(stage))")
end

"""
    V_start_buffer(stage) -> AbstractArray
    Λ_end_buffer(stage)   -> AbstractArray

Accessors for the preallocated buffers. Default to looking up the fields
`.V_start` and `.Λ_end` respectively; concrete stages can override if they
store the buffers under different names.
"""
V_start_buffer(stage::AbstractStage) = stage.V_start
Λ_end_buffer(stage::AbstractStage) = stage.Λ_end

"""CLAUDE
Allocate the V_start and Λ_end buffers for a layout-shaped stage. The
two buffers must share the same concrete array type so backward /
forward sweeps share storage shape; this helper enforces that via a
type-equality assertion. Returns a NamedTuple `(; Vs, Λe)` for
destructuring at the call site:

```julia
(; Vs, Λe) = _alloc_VΛ(layout, T, V_start, Λ_end)
```

Per-stage constructors call this in place of the repeated four-line
preamble (`dims = layout_size(layout); Vs = @something …; Λe =
@something …; @assert typeof(Vs) === typeof(Λe)`).
"""
function _alloc_VΛ(layout, ::Type{T}, V_start, Λ_end) where {T}
    dims = layout_size(layout)
    Vs   = @something V_start zeros(T, dims)
    Λe   = @something Λ_end   zeros(T, dims)
    @assert typeof(Vs) === typeof(Λe) "V_start and Λ_end must have the same concrete array type"
    return (; Vs, Λe)
end

# Dependency machinery #
#----------------------#

"""
    static_env_deps(::Type{S}) -> NamedTuple

The `env` fields the *type* `S` itself reads, regardless of user
closures. Concrete stages override; the default is `NamedTuple()`
(reads no env field).
"""
static_env_deps(::Type{<:AbstractStage}) = NamedTuple()

"""
    effective_env_slice(stage) -> NTuple{N, Symbol}

Names of `env` fields read by this stage instance. Union of two
sources:

  * `static_env_deps(typeof(stage))` — env fields baked into the stage
    implementation (none for the v1 stages: every user-visible env
    dependency flows through closures, which are not introspected).
  * `swept_key(p)` for any `Param`-typed field with `is_swept(p) == true`.

User closures (`wealth_post`, `utility`, `flow_payoff`, …) are NOT
required to declare which env fields they read; missing fields surface
as `getproperty` errors at the first backward / forward call. Used by
[`ChainStage`](@ref)'s [`chain_env_names`](@ref) for construction-time
sweep-key validation and (later) change-set propagation / F_J slicing.
"""
function effective_env_slice(stage::AbstractStage)
    static  = keys(static_env_deps(typeof(stage)))
    swept   = _swept_param_keys(stage)
    return Tuple(unique((static..., swept...)))
end

# Walk runtime fields looking for swept Params. Called once at chain
# construction, not in hot paths.
function _swept_param_keys(stage)
    syms = Symbol[]
    for fn in fieldnames(typeof(stage))
        f = getfield(stage, fn)
        if f isa Param && is_swept(f)
            push!(syms, swept_key(f)::Symbol)
        end
    end
    return Tuple(syms)
end

"""
    validate_env(stage_or_chain, env::NamedTuple) -> Nothing

Check that `env` provides every field in
`effective_env_slice(stage_or_chain)`. Throws an informative error
listing the missing keys; returns `nothing` on success.
"""
function validate_env(stage::AbstractStage, env)
    needed = effective_env_slice(stage)
    missing_keys = Symbol[]
    for k in needed
        haskey(env, k) || push!(missing_keys, k)
    end
    isempty(missing_keys) ||
        error("env is missing required fields: $(missing_keys); provided keys: $(keys(env))")
    return nothing
end
