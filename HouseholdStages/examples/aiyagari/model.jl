#####################################################
# Aiyagari (1994) — Heterogeneous-Agent Steady State #
#####################################################

# Smallest end-to-end exercise of the HouseholdStages package. The
# within-period problem decomposes into three stages, in time order:
#
#     IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings
#
# (the canonical L03 / L04 decomposition.) `IncomeShock` resolves the
# Markov draw on the income axis. `IncomeReceipt` is a deterministic
# wealth-change `b ↦ (1+r) b + w y`. `ConsumptionSavings` then chooses
# next-period wealth on the wealth grid; the implicit budget is the
# trivial `c = b_in - b_end`. Production is Cobb-Douglas with fixed
# labor, written as a plain function (no AbstractBlock /
# EquilibriumProblem machinery — those were dropped in the 2026-05-13
# refactor, see REFACTOR_PLAN.md).
#
# The wealth grid is exponentially spaced: dense near zero (where the
# borrowing constraint binds and policies are highly nonlinear) and
# coarse at the top. The 3-stage chain interpolates V linearly through
# `WealthChange.backward`, so the top of the grid must be far enough
# out that the post-receipt wealth `(1+r) b + w y` stays within the
# grid for all active cells; otherwise linear extrapolation past the
# top would amplify V by `extrap_distance / top_step`, breaking the
# Bellman contraction. An exponential grid solves this with modest N_w.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct AiyagariParams
    β :: Float64       = 0.96                  # discount factor
    σ :: Float64       = 1.5                   # CRRA risk aversion
    α :: Float64       = 0.36                  # capital share
    δ :: Float64       = 0.08                  # depreciation rate
    L :: Float64       = 1.0                   # fixed aggregate labor
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]            # idiosyncratic productivity levels
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;              # Markov transition (rows = from)
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w   :: Int       = 400                   # wealth-grid size
    w_min :: Float64   = 0.0
    w_max :: Float64   = 100.0
end

Base.Broadcast.broadcastable(p::AiyagariParams) = Ref(p)

const aiyagari_params = AiyagariParams()


# State layout #
#--------------#

"""CLAUDE
Exponentially-spaced wealth grid on `[lo, hi]` with `n` points: dense
near `lo` and coarse near `hi`. The transform `exp(t)·s - s` with
`t ∈ [0, log((hi-lo+s)/s)]` and shift `s = 1` puts the first knot at
`lo` and the last at `hi`. Used in place of an evenly-spaced grid so
the upper tail (where `WealthChange.backward` may extrapolate V past
the top knot) is wide enough that extrapolation cannot break the
Bellman contraction.
"""
function exp_wealth_grid(lo::Real, hi::Real, n::Int; shift::Real = 1.0)
    return [exp(t) * shift - shift + lo
            for t in range(0.0, log((hi - lo + shift) / shift); length = n)]
end

function aiyagari_layout(p = aiyagari_params)
    return StateLayout(
        StateAxis(:wealth, continuous_grid(exp_wealth_grid(p.w_min, p.w_max, p.N_w))),
        StateAxis(:income, discrete_finite(p.y_grid)),
    )
end


# Utility #
#---------#

_u_crra(c, ::Val{1}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)


# Household stages #
#------------------#

"Markov stage on the :income axis using transition matrix `P_y`."
function aiyagari_income_shock(layout::StateLayout, p = aiyagari_params)
    return MarkovAlong(layout; axis = :income, transition = p.P_y)
end

"""CLAUDE
Income-receipt stage: deterministic wealth update
`b_post = (1 + r) * b_pre + w * y`. Backward interpolates `V_end`
linearly along the wealth axis at the post-receipt wealth value;
forward pushes Λ via share-based redistribution.

`WealthChange` evaluates `wealth_post` via a broadcast over the cell
array with `env = Ref(env)`, so the closure receives env as a Ref and
must unwrap it with `env[]` before field access.
"""
function aiyagari_income_receipt(layout::StateLayout, p = aiyagari_params)
    function wp(cell; env)
        e = env[]                              # WealthChange passes env as Ref
        return (1 + e.r) * cell.wealth + e.w * cell.income
    end
    return WealthChange(layout;
        wealth_post  = wp,
        wealth_axis  = :wealth,
        closure_deps = (:r, :w),
    )
end

"""CLAUDE
Consumption-savings stage on the wealth grid. The household picks
next-period wealth `b_end` from the wealth grid; implied consumption is
`c = b_in - b_end` (the trivial budget inside `ConsumptionSavings`).
"""
function aiyagari_consumption_savings(layout::StateLayout, p = aiyagari_params)
    return ConsumptionSavings(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,    # concave u + linear budget ⇒ MPS ≥ 0
    )
end


# Production prices (plain function, no AbstractBlock) #
#------------------------------------------------------#

"Cobb-Douglas factor prices at aggregate capital `K`, fixed labor `p.L`."
function aiyagari_prices(K::Real, p = aiyagari_params)
    (; α, δ, L) = p
    r = α * (K / L)^(α - 1) - δ
    w = (1 - α) * (K / L)^α
    return (;r, w)
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-lifted household stage
`income_shock ∘ₛ income_receipt ∘ₛ consumption_savings` with the
`K_supplied = ∫ wealth dΛ` moment attached.
"""
function aiyagari_household(p = aiyagari_params)
    layout  = aiyagari_layout(p)
    shock   = aiyagari_income_shock(layout, p)
    receipt = aiyagari_income_receipt(layout, p)
    savings = aiyagari_consumption_savings(layout, p)
    chain   = shock ∘ₛ receipt ∘ₛ savings
    return lift_moments(chain;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end
