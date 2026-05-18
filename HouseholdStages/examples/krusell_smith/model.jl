################################################################
# Krusell-Smith (1998) — Deterministic-Aggregate Steady State #
################################################################

# Same within-period structure as Aiyagari (`../aiyagari/model.jl`),
# specialised to the K-S employed/unemployed income process. The
# household problem decomposes into three stages, in time order:
#
#     IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings
#
# `IncomeShock` resolves the two-state Markov draw on the income axis
# (employed `y = 1`, unemployed `y = 0.07` — the canonical K-S
# calibration). `IncomeReceipt` is the
# deterministic wealth update `b ↦ (1+r) b + w y`. `ConsumptionSavings`
# then chooses next-period wealth on the wealth grid with implicit
# budget `c = b_in - b_end`. Production is Cobb-Douglas with the TFP
# level `A` carried explicitly so a deterministic-aggregate driver
# (`steady_state.jl`) can sweep it; effective labor is the stationary
# distribution of the income chain.
#
# The wealth grid is exponentially spaced (dense near zero, coarse near
# the top) for the same reason as in Aiyagari: `WealthChange.backward`
# linearly extrapolates V past the top knot for cells where
# `(1+r) b + w y > w_max`, and on a uniform grid this amplifies V each
# pass and breaks the Bellman contraction. K-S calibration runs at a
# higher capital level than Aiyagari (this calibration converges to
# K = 12.88 vs Aiyagari's K = 5.69), so `w_max` is scaled up to 200
# to keep the active region well-resolved.
#
# Calibration note (β = 0.96): the textbook K-S calibration uses
# quarterly β = 0.99 with a wider wealth grid; the survey number
# "K ≈ 11-12" comes from a different calibration. The 2026-05-12 audit
# (now in `_attic/calibration_audit.jl`) found that β = 0.96
# (annual-style) with α = 0.36, δ = 0.025 lands K_RA ≈ 12.4, matching
# the canonical survey number. This example uses β = 0.96 for clarity;
# the converged steady-state K = 12.88 (at N_w = 400, 2026-05-17).

using HouseholdStages
using LinearAlgebra: I


# Parameters #
#------------#

@kwdef struct KSParams
    β :: Float64       = 0.96                  # annual-style discount factor
    γ :: Float64       = 1.0                   # log utility (CRRA with σ = 1)

    α :: Float64       = 0.36                  # capital share
    δ :: Float64       = 0.025                 # depreciation rate (quarterly value retained)

    # Idiosyncratic productivity: unemployed / employed. `y_unemp = 0.07`
    # is the canonical K-S calibration (Krusell & Smith 1998); a strictly
    # positive unemployed income is needed under log utility so the
    # b = 0 corner remains feasible (with `y_unemp = 0` the consumption-
    # savings stage's V_start[b = 0, y_unemp] is -Inf, which cascades
    # through the linear V-interpolation in `WealthChange.backward` and
    # breaks the VFI).
    y_grid :: Vector{Float64} = [0.07, 1.0]
    P_y    :: Matrix{Float64} = [0.6   0.4;
                                 0.05  0.95]

    # Wealth grid (exponential). `w_max = 200`, `N_w = 100` is sized so
    # `(1+r)*w_max + w*y_emp < w_max + top_step` at the equilibrium
    # `r ≈ 0.04`, keeping `WealthChange.backward`'s top-end linear
    # extrapolation well below the contraction bound. The grid is wider
    # and finer than Aiyagari's because K-S sits closer to the
    # impatience watershed `β(1+r) = 1`, where the household saves
    # aggressively in response to small changes in r.
    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 200.0
end

Base.Broadcast.broadcastable(p::KSParams) = Ref(p)

const ks_params = KSParams()


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

function ks_layout(p = ks_params)
    return StateLayout(
        StateAxis(:wealth, continuous_grid(exp_wealth_grid(p.w_min, p.w_max, p.N_w))),
        StateAxis(:income, discrete_finite(p.y_grid)),
    )
end


# Utility #
#---------#

# CRRA with the σ = 1 (log) branch handled via `Val{σ}` dispatch — note
# the explicit `Val{1.0}` method, because `Val(1.0)` does not dispatch
# to `Val{1}` (the type parameter is the Float, not the Int). The
# generic branch `(c^(1-σ))/(1-σ)` would yield `Inf/0` at σ = 1.
_u_crra(c, ::Union{Val{1}, Val{1.0}}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)


# Effective labor (stationary distribution over income) #
#-------------------------------------------------------#

"""CLAUDE
Stationary-distribution-weighted aggregate labor for the income chain
`P_y`: solve `π' P_y = π'` (with `Σπ = 1`) and return `Σ π_i y_i`. Used
by `ks_prices` so the Cobb-Douglas wage and rental rate reflect the
true effective labor supply when the income process has a non-trivial
employed/unemployed split.
"""
function ks_effective_labor(P_y::AbstractMatrix, y_grid::AbstractVector)
    n = size(P_y, 1)
    A = P_y' - I(n)
    A[end, :] .= 1.0
    rhs = zeros(n); rhs[end] = 1.0
    π = A \ rhs
    return sum(y_grid .* π)
end


# Household stages #
#------------------#

"Markov stage on the :income axis using transition matrix `P_y`."
function ks_income_shock(layout::StateLayout, p = ks_params)
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
function ks_income_receipt(layout::StateLayout, p = ks_params)
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
Log utility (γ = 1) is the K-S baseline; the param wrapper supports
non-unit CRRA by setting `p.γ`.
"""
function ks_consumption_savings(layout::StateLayout, p = ks_params)
    return ConsumptionSavings(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.γ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
end


# Production prices (plain function, no AbstractBlock) #
#------------------------------------------------------#

"Cobb-Douglas factor prices at aggregate capital `K` and TFP `A`."
function ks_prices(K::Real, A::Real, p = ks_params)
    (; α, δ) = p
    L = ks_effective_labor(p.P_y, p.y_grid)
    r = α * A * (K / L)^(α - 1) - δ
    w = (1 - α) * A * (K / L)^α
    return (;r, w)
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-lifted household stage
`income_shock ∘ₛ income_receipt ∘ₛ consumption_savings` with the
`K_supplied = ∫ wealth dΛ` moment attached.
"""
function ks_household(p = ks_params)
    layout  = ks_layout(p)
    shock   = ks_income_shock(layout, p)
    receipt = ks_income_receipt(layout, p)
    savings = ks_consumption_savings(layout, p)
    chain   = shock ∘ₛ receipt ∘ₛ savings
    return lift_moments(chain;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end
