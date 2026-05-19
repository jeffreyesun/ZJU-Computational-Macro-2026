"""CLAUDE
    AssetPriceChangeStage(layout; holdings_axis, wealth_axis = :wealth,
                     q_field = :q, q_last_field = :q_last,
                     extrap = :linear,
                     element_type = Float64,
                     V_start = nothing, Λ_end = nothing) -> WealthChangeStage

Convenience constructor for the "asset price revaluation" stage: existing
holders of an asset on `holdings_axis` gain `(env.q - env.q_last) *
cell.holdings_axis` on their wealth.

Returns a [`WealthChangeStage`](@ref) instance whose `wealth_post` closure is

```julia
(cell; env) -> getfield(cell, wealth_axis) +
               (getfield(env, q_field) - getfield(env, q_last_field)) *
               getfield(cell, holdings_axis)
```

(`env` is passed plainly; the closure reads its fields directly.) The
returned object is a plain `WealthChangeStage` (all `WealthChangeStage`
methods — `allocate`, `backward!`, `forward!`, `with_eltype`,
`lift_gpu`, the forward-mode AD path — apply unchanged); this
constructor is sugar for the common case, not a parallel implementation.

# Example

```julia
layout = StateLayout(
    StateAxis(:wealth, continuous_grid(0.0, 50.0; length = 64)),
    StateAxis(:h,      [0.0, 1.0, 2.0]),       # housing stock
)
stage = AssetPriceChangeStage(layout; holdings_axis = :h)
env   = (q = 1.05, q_last = 1.0)
buffers = allocate(stage)
backward!(stage, V_post, env, buffers)
```
"""
function AssetPriceChangeStage(layout::StateLayout;
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
    return WealthChangeStage(layout;
                        wealth_post  = wealth_post,
                        wealth_axis  = wealth_axis,
                        extrap       = extrap,
                        element_type = element_type,
                        V_start, Λ_end)
end

"""CLAUDE
Build the per-cell wealth-revaluation closure used by
[`AssetPriceChangeStage`](@ref). Factored out so the closure body is named
and traceable; the returned function is a plain Julia closure that
closes over the four `Symbol`s (axis / field names) by capture.

`env` is passed directly (no `Ref` wrapper), so the closure reads
`env.q` / `env.q_last` and `cell.wealth_axis` / `cell.holdings_axis`
without unpacking.
"""
function _make_asset_price_closure(wealth_axis::Symbol, holdings_axis::Symbol,
                                   q_field::Symbol, q_last_field::Symbol)
    return function (cell; env)
        w   = getfield(cell, wealth_axis)
        h   = getfield(cell, holdings_axis)
        q   = getfield(env,  q_field)
        q⁻  = getfield(env,  q_last_field)
        return w + (q - q⁻) * h
    end
end
