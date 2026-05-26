"""
Specification for a single moment to emit from a household chain.

The `integrand` is a `(cell; env)` closure returning the per-cell
integrand value. `reduce` is applied to the per-cell mass-weighted
integrand (typically `sum` or `mean`). `aggregate_over` and `per`
(both `Symbol` or `nothing`) select a product axis to reduce over or
split along; `weights` (`Symbol` or `nothing`) names an env field or
stage-kernel weight (`nothing` means Λ is the weight).

Construct via [`at_end`](@ref) (typical).
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
not at construction time.
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
    define_moment!(chain, name::Symbol, spec::MomentSpec;
                   overwrite_existing_moment_definitions = false) -> chain

Attach a moment named `name` to the chain. The moments live in a
mutable `Dict` on the chain's `Spec` so subsequent calls can extend
the chain's moment menu without rebuilding it.

Append-only by default: redefining an existing name errors. Pass
`overwrite_existing_moment_definitions = true` for the rare
legitimate-overwrite case. The verbose kwarg name is deliberate —
silent shadowing is hard to debug, and this opts the user in
explicitly.

Returns the chain (mutated), so the form
`hh = define_moment!(hh, ...)` reads cleanly.
"""
function define_moment!(chain::ChainStage, name::Symbol, spec::MomentSpec;
                        overwrite_existing_moment_definitions::Bool = false)
    m = chain.spec.moments
    if haskey(m, name) && !overwrite_existing_moment_definitions
        error("define_moment!: moment :$name is already defined on this chain. " *
              "Pass overwrite_existing_moment_definitions = true to overwrite, " *
              "or rebuild the chain.")
    end
    m[name] = spec
    return chain
end

"""
    define_moments!(chain; kwargs..., overwrite_existing_moment_definitions = false)
        -> chain

Batch form. Each kwarg is `name = MomentSpec(...)`. Same append-only
semantics as [`define_moment!`](@ref).
"""
function define_moments!(chain::ChainStage;
                         overwrite_existing_moment_definitions::Bool = false,
                         kwargs...)
    for (name, spec) in kwargs
        define_moment!(chain, name, spec;
                       overwrite_existing_moment_definitions)
    end
    return chain
end

# A single stage can be promoted to a singleton chain to receive moments.
define_moment!(stage::AbstractStage, name::Symbol, spec::MomentSpec; kwargs...) =
    define_moment!(ChainStage((stage,)), name, spec; kwargs...)
define_moments!(stage::AbstractStage; kwargs...) =
    define_moments!(ChainStage((stage,)); kwargs...)

# Moment computation #
#--------------------#

"""
    compute_moments(spec::ChainStageSpec, Λ, env) -> NamedTuple
    compute_moments(chain::ChainStage,    Λ, env) -> NamedTuple

Evaluate every moment spec attached to the chain against the given
distribution `Λ` and `env`. Returns a NamedTuple keyed by spec
name. A spec without `aggregate_over` / `per` is reduced to a scalar
via `spec.reduce(per_cell_array)`, where the per-cell array is
`integrand * mass`.

Errors if the chain has no moment specs attached.

The Λ is passed explicitly (the function does not reach into any
buffer state). The chain's output layout is read from the Spec to
find cell coordinates.

The Spec-keyed signature is the primary; the bundled-stage form is
one-line sugar.
"""
function compute_moments(spec::ChainStageSpec, layout::StateLayout, Λ, env)
    moments = spec.moments
    @assert !isempty(moments) "compute_moments: ChainStageSpec has no moments attached; call define_moment! first."
    out = Dict{Symbol, Any}()
    for (name, mspec) in moments
        out[name] = _eval_spec(mspec, layout, Λ, env)
    end
    return NamedTuple{Tuple(keys(out))}(Tuple(values(out)))
end

# Compat: spec-only call uses the spec's terminal output layout (chain walks components).
compute_moments(spec::ChainStageSpec, Λ, env) =
    error("compute_moments(spec, Λ, env): need a layout. Call `compute_moments(stage, Λ, env)` or `compute_moments(spec, layout, Λ, env)`.")

compute_moments(chain::ChainStage, Λ, env) =
    compute_moments(chain.spec, chain.buffer.output_layout, Λ, env)

function _eval_spec(spec::MomentSpec, layout::StateLayout, Λ, env)
    cells_arr = cell_array(layout)
    return spec.reduce(spec.integrand.(cells_arr; env) .* Λ)
end
