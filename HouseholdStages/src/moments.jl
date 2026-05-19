"""
Specification for a single moment to emit from a household chain.

The `integrand` is a `(cell; env)` closure returning the per-cell
integrand value. `reduce` is applied to the per-cell mass-weighted
integrand (typically `sum` or `mean`). `aggregate_over` and `per`
(both `Symbol` or `nothing`) select a product axis to reduce over or
split along; `weights` (`Symbol` or `nothing`) names an env field or
stage-kernel weight (`nothing` means Λ is the weight).

Construct via [`at_end`](@ref) (typical) or `moment` (anchored to a
specific inner stage; Step 14).
"""
struct MomentSpec{F, R, AO, W, P}
    integrand      :: F
    reduce         :: R
    aggregate_over :: AO
    weights        :: W
    per            :: P
end

"""
    at_end(; integrand, reduce, aggregate_over=nothing, weights=nothing,
                  per=nothing) -> MomentSpec

Construct a moment anchored at the end of the household chain.

`integrand` is a closure `(cell; env) -> value` returning the per-cell
integrand. As a convenience, a `Symbol` is interpreted as a cell-field
shortcut: `:wealth` is sugar for `(cell; env) -> cell.wealth`. `env`
fields the closure reads are validated at runtime (by `getproperty`),
not at construction time — the package does not require an explicit
dependency declaration.
"""
function at_end(; integrand,
                reduce,
                aggregate_over = nothing,
                weights        = nothing,
                per            = nothing)
    fp = _wrap_integrand(integrand)
    return MomentSpec(fp, reduce, aggregate_over, weights, per)
end

# Integrand may be a closure or a `Symbol` cell-field shortcut.
_wrap_integrand(f) = f
_wrap_integrand(s::Symbol) = (cell; env) -> getproperty(cell, s)

"""
    lift_moments(stage; specs...) -> ChainStage

Return a new `ChainStage` with the given moment `specs` attached. If
`stage` is already a `ChainStage`, the new chain carries the same stages
and the new moments (errors if the input already has moments); if
`stage` is a single stage, it is wrapped into a singleton chain.

After a forward pass, [`compute_moments`](@ref) evaluates every spec
against the chain's terminal `Λ_end`.
"""
function lift_moments(stage::AbstractStage; specs...)
    chain = stage isa ChainStage ? stage : ChainStage((stage,))
    isempty(chain.moments) ||
        error("lift_moments: chain already has moments attached; lift them once, last.")
    nt = NamedTuple{Tuple(keys(specs))}(Tuple(values(specs)))
    return ChainStage(chain.stages; moments = nt)
end

# Moment computation #
#--------------------#

"""
    compute_moments(chain::ChainStage, env) -> NamedTuple

Evaluate every spec in `chain.moments` against the chain's terminal
`Λ_end` and `env`. Returns a NamedTuple keyed by spec name. A spec
without `aggregate_over` / `per` is reduced to a scalar via
`spec.reduce(per_cell_array)`, where the per-cell array is
`integrand * mass`. `reduce = sum` and `reduce = mean` are the common
cases.

Errors if the chain has no moment specs attached.
"""
function compute_moments(chain::ChainStage, env)
    isempty(chain.moments) &&
        error("compute_moments: ChainStage has no moments attached; call lift_moments first.")
    Λ = Λ_end_buffer(chain)
    return _compute_each(chain.moments, chain.out_layout, Λ, env)
end

# Walk the NamedTuple of specs and produce a NamedTuple of values.
@generated function _compute_each(specs::NamedTuple{Names},
                                  layout, Λ, env) where {Names}
    exprs = [:(_eval_spec(specs.$n, layout, Λ, env)) for n in Names]
    return Expr(:tuple, Expr(:parameters, [Expr(:kw, n, exprs[i]) for (i, n) in enumerate(Names)]...))
end

function _eval_spec(spec::MomentSpec, layout::StateLayout, Λ, env)
    cells_arr = cell_array(layout)
    return spec.reduce(spec.integrand.(cells_arr; env) .* Λ)
end
