#####################################################
# Spatial Aiyagari — Two Locations, Migration via LogitChoice #
#####################################################

# Smallest spatial extension of the Aiyagari household. The state space
# carries a third axis `:location` over `[:home, :abroad]`, and the
# within-period problem decomposes into four stages, in time order:
#
#     IncomeShock ∘ₛ LogitChoiceMigration ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings
#
# The decomposition for wealth + income is the canonical L03/L04 form
# (same as `../aiyagari/model.jl`); the migration stage is inserted
# *after* the income shock and *before* income receipt. Households see
# their income draw, then decide whether to move (logit-smoothed with a
# migration cost), and finally receive location-specific income and
# choose savings. `LogitChoice` acts on the `:location` axis with action
# set `{:home, :abroad}` and stores the destination-choice probability
# tensor. Conditional on the new location, `WealthChange` reads the
# location-specific `(r, w)` from env and applies the deterministic
# wealth update.
#
# The wealth grid is exponentially spaced: dense near zero (binding
# borrowing constraint) and coarse at the top. Same reason as the other
# examples — `WealthChange.backward` linearly extrapolates V past the
# top knot, and a uniform grid amplifies V each pass and breaks the
# Bellman contraction. The two locations clear their capital markets
# independently in the outer damped Picard on `(K_home, K_abroad)`; no
# inter-location trade.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct SpatialParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36
    δ :: Float64       = 0.08
    L :: Float64       = 1.0                              # total labor (split L/2 per location)
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 30.0
    # Location-specific TFP — kept equal for the baseline. A non-zero
    # productivity gap induces period-3 oscillations under damped Picard
    # on `(K_home, K_abroad)` (documented in the README); Anderson
    # acceleration or Newton would handle asymmetric calibrations.
    A_home    :: Float64 = 1.0
    A_abroad  :: Float64 = 1.0
    # Logit scale for the location choice. Large ε keeps the choice
    # close to uniform conditional on `migration_cost`, which stabilizes
    # the 2-D fixed point under damped Picard.
    ε_logit         :: Float64 = 5.0
    migration_cost  :: Float64 = 0.5
end

Base.Broadcast.broadcastable(p::SpatialParams) = Ref(p)

const params = SpatialParams()


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

function spatial_layout(p = params)
    return StateLayout(
        StateAxis(:wealth,   continuous_grid(exp_wealth_grid(p.w_min, p.w_max, p.N_w))),
        StateAxis(:income,   discrete_finite(p.y_grid)),
        StateAxis(:location, categorical([:home, :abroad])),
    )
end


# Utility #
#---------#

# CRRA with the σ = 1 (log) branch handled via `Val{σ}` dispatch. Both
# `Val{1}` (Int) and `Val{1.0}` (Float64) dispatch to log to defend
# against the calibration writing `σ = 1.0` as a Float64.
_u_crra(c, ::Union{Val{1}, Val{1.0}}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)


# Household stages #
#------------------#

"Markov stage on the :income axis using transition matrix `P_y`."
function spatial_income_shock(layout::StateLayout, p = params)
    return MarkovAlong(layout; axis = :income, transition = p.P_y)
end

"""CLAUDE
Migration stage on the `:location` axis. Households compare staying vs.
moving via a logit-smoothed discrete choice with action set
`{:home, :abroad}`. The cost matrix has zero on the diagonal and
`p.migration_cost` off-diagonal — symmetric two-location cost.

The dedicated [`Migration`](@ref) stage carries this cost matrix as
data, so no `flow_payoff` closure is needed (and `migration_cost`
no longer needs to live on `env`). The logit scale `ε = p.ε_logit` is
kept high in the baseline so the choice is close to uniform conditional
on `migration_cost`, which stabilizes the outer 2-D Picard.
"""
function spatial_migration(layout::StateLayout, p = params)
    C = [0.0           p.migration_cost;
         p.migration_cost 0.0]
    return Migration(layout;
        location_axis  = :location,
        migration_cost = C,
        ε              = Param(p.ε_logit),
    )
end

"""CLAUDE
Income-receipt stage: deterministic wealth update
`b_post = (1 + r_loc) * b_pre + w_loc * y`, where `(r_loc, w_loc)` is
the location-specific price pair drawn from env according to the
post-migration location. Backward interpolates `V_end` linearly along
the wealth axis at the post-receipt wealth value; forward pushes Λ via
share-based redistribution.

`WealthChange` evaluates `wealth_post` via a broadcast over the cell
array with `env = Ref(env)`, so the closure receives env as a Ref and
must unwrap it with `env[]` before field access.
"""
function spatial_income_receipt(layout::StateLayout, p = params)
    function wp(cell; env)
        e = env[]
        r_loc = cell.location == :home ? e.r_home : e.r_abroad
        w_loc = cell.location == :home ? e.w_home : e.w_abroad
        return (1 + r_loc) * cell.wealth + w_loc * cell.income
    end
    return WealthChange(layout;
        wealth_post  = wp,
        wealth_axis  = :wealth,
        closure_deps = (:r_home, :r_abroad, :w_home, :w_abroad),
    )
end

"""CLAUDE
Consumption-savings stage on the wealth grid. The household picks
next-period wealth `b_end` from the wealth grid; implied consumption is
`c = b_in - b_end` (the trivial budget inside `ConsumptionSavings`).
"""
function spatial_consumption_savings(layout::StateLayout, p = params)
    return ConsumptionSavings(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
end


# Production prices per location (plain function, no AbstractBlock) #
#-------------------------------------------------------------------#

"""
Cobb-Douglas factor prices at each location. Aggregate labor `p.L` is
split equally between the two locations (`L_each = L / 2`); each
location's capital market clears independently in the outer loop.
"""
function spatial_prices(K_home::Real, K_abroad::Real, p = params)
    (; α, δ, L, A_home, A_abroad) = p
    L_each   = L / 2
    r_home   = α * A_home * (K_home / L_each)^(α - 1) - δ
    w_home   = (1 - α) * A_home * (K_home / L_each)^α
    r_abroad = α * A_abroad * (K_abroad / L_each)^(α - 1) - δ
    w_abroad = (1 - α) * A_abroad * (K_abroad / L_each)^α
    return (; r_home, w_home, r_abroad, w_abroad)
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-lifted household stage
`income_shock ∘ₛ migration ∘ₛ income_receipt ∘ₛ consumption_savings`
with per-location capital and population moments lifted at the end.
"""
function spatial_household(p = params)
    layout  = spatial_layout(p)
    shock   = spatial_income_shock(layout, p)
    move    = spatial_migration(layout, p)
    receipt = spatial_income_receipt(layout, p)
    savings = spatial_consumption_savings(layout, p)
    chain   = shock ∘ₛ move ∘ₛ receipt ∘ₛ savings
    return lift_moments(chain;
        K_home     = at_end(
            integrand = (cell; env) -> cell.location == :home ? cell.wealth : 0.0,
            reduce = sum),
        K_abroad   = at_end(
            integrand = (cell; env) -> cell.location == :abroad ? cell.wealth : 0.0,
            reduce = sum),
        pop_home   = at_end(
            integrand = (cell; env) -> cell.location == :home ? 1.0 : 0.0,
            reduce = sum),
        pop_abroad = at_end(
            integrand = (cell; env) -> cell.location == :abroad ? 1.0 : 0.0,
            reduce = sum),
    )
end
