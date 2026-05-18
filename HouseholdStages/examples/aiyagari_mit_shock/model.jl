#####################################################
# Aiyagari with MIT Shock — Model Definition         #
#####################################################

# Same within-period structure as Aiyagari (`../aiyagari/model.jl`),
# specialised to a transition path under a one-time unanticipated TFP
# shock with AR(1) decay. The household problem decomposes into three
# stages, in time order:
#
#     IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings
#
# (the canonical L03 / L04 decomposition.) `IncomeShock` resolves the
# Markov draw on the income axis. `IncomeReceipt` is a deterministic
# wealth-change `b ↦ (1+r) b + w y`. `ConsumptionSavings` then chooses
# next-period wealth on the wealth grid; the implicit budget is the
# trivial `c = b_in - b_end`.
#
# What is MIT-specific. Production is Cobb-Douglas in `(K_t, A_t)` with
# a fixed labor `L`; the TFP `A_t` is carried as an explicit argument
# to `aiyagari_prices` (rather than baked into the chain) so the
# transition driver can sweep it period by period. The env consumed by
# the household chain itself stays minimal — `(;K, r, w)` — exactly
# matching the steady-state example; `A` enters only through prices.
#
# Per the project's per-example self-contained discipline (see
# `HouseholdStages/examples/PLAN.md`), the model code is duplicated rather than
# imported from `../aiyagari/model.jl`. The wealth grid is exponentially
# spaced for the same reason as in Aiyagari: `WealthChange.backward`
# linearly extrapolates V past the top knot for cells where
# `(1+r) b + w y > w_max`, and on a uniform grid this amplifies V each
# pass and breaks the Bellman contraction.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MITShockParams
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

Base.Broadcast.broadcastable(p::MITShockParams) = Ref(p)

const mit_shock_params = MITShockParams()


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

function aiyagari_layout(p = mit_shock_params)
    return StateLayout(
        StateAxis(:wealth, continuous_grid(exp_wealth_grid(p.w_min, p.w_max, p.N_w))),
        StateAxis(:income, discrete_finite(p.y_grid)),
    )
end


# Utility #
#---------#

_u_crra(c, ::Union{Val{1}, Val{1.0}}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)


# Household stages #
#------------------#

"Markov stage on the :income axis using transition matrix `P_y`."
function aiyagari_income_shock(layout::StateLayout, p = mit_shock_params)
    return MarkovAlong(layout; axis = :income, transition = p.P_y)
end

"""CLAUDE
Income-receipt stage: deterministic wealth update
`b_post = (1 + r) * b_pre + w * y`. Backward interpolates `V_end`
linearly along the wealth axis at the post-receipt wealth value;
forward pushes Λ via share-based redistribution.

`WealthChange` evaluates `wealth_post` via a broadcast over the cell
array with `env = Ref(env)`, so the closure receives env as a Ref and
must unwrap it with `env[]` before field access. The TFP path `A_t`
is folded into prices upstream by `aiyagari_prices`; this closure
reads only `r, w` from env.
"""
function aiyagari_income_receipt(layout::StateLayout, p = mit_shock_params)
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
function aiyagari_consumption_savings(layout::StateLayout, p = mit_shock_params)
    return ConsumptionSavings(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
end


# Production prices (plain function, no AbstractBlock) #
#------------------------------------------------------#

"""
Cobb-Douglas factor prices at aggregate capital `K`, fixed labor `p.L`,
and TFP `A`. `A` is explicit so the transition driver can sweep it
period by period while keeping the household chain's env minimal.
"""
function aiyagari_prices(K::Real, A::Real, p = mit_shock_params)
    (; α, δ, L) = p
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
function aiyagari_household(p = mit_shock_params)
    layout  = aiyagari_layout(p)
    shock   = aiyagari_income_shock(layout, p)
    receipt = aiyagari_income_receipt(layout, p)
    savings = aiyagari_consumption_savings(layout, p)
    chain   = shock ∘ₛ receipt ∘ₛ savings
    return lift_moments(chain;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# TFP shock path #
#----------------#

"""CLAUDE
Deterministic AR(1) decay of TFP from `A_0` back to steady state
`A_ss`. Returns a length-`T` vector with `A[1] = A_0` and
`A[t] = A_ss + ρ (A[t-1] - A_ss)` for `t ≥ 2`. Defined locally because
the library deliberately leaves transition-side utilities to the
consumer (see `PROJECT_PLAN.md` Decisions log 2026-05-18).
"""
function tfp_path(T::Int; A_0::Float64 = 1.05, ρ::Float64 = 0.85,
                  A_ss::Float64 = 1.0)
    A = zeros(T)
    A[1] = A_0
    for t in 2:T
        A[t] = A_ss + ρ * (A[t-1] - A_ss)
    end
    return A
end
