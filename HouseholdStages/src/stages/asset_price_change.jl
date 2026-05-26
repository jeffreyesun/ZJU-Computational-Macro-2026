"""
Convenience constructor for the asset-revaluation pattern: existing
holders of an asset on `holdings_axis` gain `(env.q - env.q_last) *
cell.holdings_axis` on their wealth. Returns a plain
[`WealthChangeStage`](@ref).
"""
function AssetPriceChangeStage(layout::StateLayout;
                               holdings_axis::Symbol,
                               wealth_axis::Symbol=:wealth,
                               q_field::Symbol=:q,
                               q_last_field::Symbol=:q_last,
                               extrap=:linear)
    wealth_post = let wa=wealth_axis, ha=holdings_axis, qf=q_field, qlf=q_last_field
        (cell; env) -> getfield(cell, wa) +
                       (getfield(env, qf) - getfield(env, qlf)) * getfield(cell, ha)
    end
    return WealthChangeStage(layout; wealth_post, wealth_axis, extrap)
end
