"""CLAUDE
    AssetPriceChange(layout; holdings_axis, wealth_axis = :wealth,
                     q_field = :q, q_last_field = :q_last,
                     extrap = :linear,
                     element_type = Float64,
                     V_start = nothing, Λ_end = nothing) -> WealthChange

Convenience constructor for the "asset price revaluation" stage: existing
holders of an asset on `holdings_axis` gain `(env[q_field] - env[q_last_field]) *
cell[holdings_axis]` on their wealth.

Returns a [`WealthChange`](@ref) instance whose `wealth_post` closure is

```julia
(cell; env) -> getfield(cell, wealth_axis) +
               (getfield(env, q_field) - getfield(env, q_last_field)) *
               getfield(cell, holdings_axis)
```

and whose `closure_deps` is `(q_field, q_last_field)`. The returned
object is a plain `WealthChange` (all `WealthChange` methods —
`allocate`, `backward!`, `forward!`, `with_eltype`, `lift_gpu`, the
forward-mode AD path — apply unchanged); this constructor is sugar for
the common case, not a parallel implementation.

# Example

```julia
layout = StateLayout(
    StateAxis(:wealth, continuous_grid(0.0, 50.0; size = 64)),
    StateAxis(:h,      discrete_finite([0.0, 1.0, 2.0])),  # housing stock
)
stage = AssetPriceChange(layout; holdings_axis = :h)
# env must carry `q` and `q_last`:
env = (q = 1.05, q_last = 1.0)
kernel, scratch = allocate(stage)
backward!(stage, V_post, env, kernel, scratch)
```
"""
function AssetPriceChange(layout::StateLayout;
                          holdings_axis::Symbol,
                          wealth_axis::Symbol = :wealth,
                          q_field::Symbol = :q,
                          q_last_field::Symbol = :q_last,
                          extrap = :linear,
                          element_type::Type{T} = Float64,
                          V_start::Union{Nothing, AbstractArray} = nothing,
                          Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T<:Real}
    wealth_post = _make_asset_price_closure(wealth_axis, holdings_axis,
                                            q_field, q_last_field)
    return WealthChange(layout;
                        wealth_post  = wealth_post,
                        wealth_axis  = wealth_axis,
                        closure_deps = (q_field, q_last_field),
                        extrap       = extrap,
                        element_type = element_type,
                        V_start, Λ_end)
end

"""CLAUDE
Build the per-cell wealth-revaluation closure used by
[`AssetPriceChange`](@ref). Factored out so the closure body is named
and traceable; the returned function is a plain Julia closure that
closes over the four `Symbol`s (axis / field names) by capture.

`WealthChange` evaluates `wealth_post` via a broadcast that passes
`env = Ref(env)`, so the closure unwraps with `env[]` before field
access — the established convention across all example consumers.
"""
function _make_asset_price_closure(wealth_axis::Symbol, holdings_axis::Symbol,
                                   q_field::Symbol, q_last_field::Symbol)
    return function (cell; env)
        e   = env[]
        w   = getfield(cell, wealth_axis)
        h   = getfield(cell, holdings_axis)
        q   = getfield(e,    q_field)
        q⁻  = getfield(e,    q_last_field)
        return w + (q - q⁻) * h
    end
end
