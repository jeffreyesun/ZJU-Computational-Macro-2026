"""
Specification for a single moment to emit from a household chain.

The `integrand` is a `(cell; env)` closure returning the per-cell
integrand value. `reduce` is applied to the per-cell mass-weighted
integrand (typically `sum` or `mean`). `aggregate_over` and `per`
(both `Symbol` or `nothing`) select a product axis to reduce over or
split along; `weights` (`Symbol` or `nothing`) names an env field or
stage-cache weight (`nothing` means Λ is the weight). `closure_deps`
lists env fields the integrand reads and is unioned into the chain's
effective env-slice.

Construct via [`at_end`](@ref) (typical) or `moment` (anchored to a
specific inner stage; Step 14).
"""
struct MomentSpec{F, R, AO, W, P, D}
    integrand      :: F
    reduce         :: R
    aggregate_over :: AO
    weights        :: W
    per            :: P
    closure_deps   :: NTuple{D, Symbol}
end

"""
    at_end(; integrand, reduce, aggregate_over=nothing, weights=nothing,
                  per=nothing, closure_deps=()) -> MomentSpec

Construct a moment anchored at the end of the household chain.

`integrand` is a closure `(cell; env) -> value` returning the per-cell
integrand. As a convenience, a `Symbol` is interpreted as a cell-field
shortcut: `:wealth` is sugar for `(cell; env) -> cell.wealth`.
"""
function at_end(; integrand,
                reduce,
                aggregate_over = nothing,
                weights        = nothing,
                per            = nothing,
                closure_deps::NTuple{D, Symbol} = ()) where {D}
    fp = _wrap_integrand(integrand)
    return MomentSpec(fp, reduce, aggregate_over, weights, per, closure_deps)
end

# Integrand may be a closure or a `Symbol` cell-field shortcut.
_wrap_integrand(f) = f
_wrap_integrand(s::Symbol) = (cell; env) -> getproperty(cell, s)

"""
    MomentedChain{Inner<:AbstractStage, Specs<:NamedTuple, L<:StateLayout} <: AbstractStage

A chain wrapped with moment-emission specs. Behaves as an `AbstractStage`
(delegates `allocate`, `backward!`, `forward!` to the inner chain). After
each forward pass, [`compute_moments`](@ref) evaluates all specs against
the chain's terminal `Λ_end` and returns a NamedTuple keyed by spec name.
"""
struct MomentedChain{Inner<:AbstractStage,
                     Specs<:NamedTuple,
                     L<:StateLayout} <: AbstractStage
    inner      :: Inner
    specs      :: Specs
    out_layout :: L
end

"""
    lift_moments(chain; specs...) -> MomentedChain

Wrap `chain` with a NamedTuple of moment specs. Each spec name becomes
a field in the moments record returned by [`compute_moments`](@ref).
"""
function lift_moments(chain::AbstractStage; specs...)
    last_stage = chain isa StageChain ? chain.stages[end] : chain
    out_layout = _stage_out_layout(last_stage)
    nt = NamedTuple{Tuple(keys(specs))}(Tuple(values(specs)))
    return MomentedChain(chain, nt, out_layout)
end

# Default accessor for a stage's output layout. Override if the stage
# uses a different field name.
function _stage_out_layout(stage::AbstractStage)
    hasfield(typeof(stage), :output_layout) || error(
        "lift_moments: terminal stage of type $(typeof(stage)) has no " *
        "output_layout field; cannot infer the moment-integration layout.")
    return stage.output_layout
end

# Inherited stage interface #
#---------------------------#

allocate(mc::MomentedChain, T::Type = Float64) = allocate(mc.inner, T)

V_start_buffer(mc::MomentedChain) = V_start_buffer(mc.inner)
Λ_end_buffer(mc::MomentedChain)   = Λ_end_buffer(mc.inner)

backward!(mc::MomentedChain, V_end, env, kernel, scratch) =
    backward!(mc.inner, V_end, env, kernel, scratch)

forward!(mc::MomentedChain, Λ_start, kernel, scratch, moments = nothing) =
    forward!(mc.inner, Λ_start, kernel, scratch, moments)

# Env-slice composition #
#-----------------------#

function effective_env_slice(mc::MomentedChain)
    names = Symbol[]
    for k in effective_env_slice(mc.inner)
        push!(names, k)
    end
    for (_, spec) in pairs(mc.specs)
        for k in spec.closure_deps
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end

function validate_env(mc::MomentedChain, env)
    needed = effective_env_slice(mc)
    missing_keys = Symbol[]
    for k in needed
        haskey(env, k) || push!(missing_keys, k)
    end
    isempty(missing_keys) ||
        error("env is missing required fields: $(missing_keys); provided keys: $(keys(env))")
    return nothing
end

# Moment computation #
#--------------------#

"""
Evaluate every spec in `mc.specs` against the chain's terminal `Λ_end`
and `env`. Returns a NamedTuple keyed by spec name. A spec without
`aggregate_over` / `per` is reduced to a scalar via
`spec.reduce(per_cell_array)`, where the per-cell array is
`integrand * mass`. `reduce = sum` and `reduce = mean` are the common
cases.
"""
function compute_moments(mc::MomentedChain, env)
    Λ = Λ_end_buffer(mc.inner)
    layout = mc.out_layout
    return _compute_each(mc.specs, layout, Λ, env)
end

# Walk the NamedTuple of specs and produce a NamedTuple of values.
@generated function _compute_each(specs::NamedTuple{Names},
                                  layout, Λ, env) where {Names}
    exprs = [:(_eval_spec(specs.$n, layout, Λ, env)) for n in Names]
    return Expr(:tuple, Expr(:parameters, [Expr(:kw, n, exprs[i]) for (i, n) in enumerate(Names)]...))
end

function _eval_spec(spec::MomentSpec, layout::StateLayout, Λ, env)
    # Build the per-cell mass-weighted integrand array.
    T = eltype(Λ)
    weighted = Array{T}(undef, size(Λ))
    for (idx, cell) in cells(layout)
        ci = CartesianIndex(values(idx))
        v  = spec.integrand(cell; env = env)
        weighted[ci] = v * Λ[ci]
    end
    return spec.reduce(weighted)
end
