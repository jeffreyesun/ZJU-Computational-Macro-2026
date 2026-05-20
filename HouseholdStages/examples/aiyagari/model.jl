#####################################################
# Aiyagari (1994) — Heterogeneous-Agent Steady State #
#####################################################

# Smallest end-to-end exercise of the HouseholdStages package. The
# within-period problem decomposes into three stages, in time order:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# (the canonical L03 / L04 decomposition.) `IncomeShock` resolves the
# Markov draw on the income axis. `IncomeReceipt` is a deterministic
# wealth-change `b ↦ (1+r) b + w y`. `ConsumptionSavingsStage` then chooses
# next-period wealth on the wealth grid; the implicit budget is the
# trivial `c = b_in - b_end`. Production is Cobb-Douglas with fixed
# labor, written as a plain function.
#
# The wealth grid is log-spaced (`continuous_grid(...; spacing = :log)`):
# dense near zero (where the borrowing constraint binds and policies are
# highly nonlinear) and coarse at the top. `WealthChangeStage.backward`
# linearly interpolates V — the top of the grid must be far enough out
# that the post-receipt wealth `(1+r) b + w y` stays within the grid for
# all active cells, otherwise extrapolation past the top would amplify V
# by `extrap_distance / top_step` and break the Bellman contraction. A
# log grid solves this with modest `N_w`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct AiyagariParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36
    δ :: Float64       = 0.08
    L :: Float64       = 1.0
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 100.0
end

Base.Broadcast.broadcastable(p::AiyagariParams) = Ref(p)

const aiyagari_params = AiyagariParams()


# Utility #
#---------#

_u_crra(c, ::Val{1}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Aiyagari household block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage` with the
`K_supplied = ∫ wealth dΛ` moment attached. The wealth-axis log grid
and the three-stage layout are inlined here.
"""
function aiyagari_household(p = aiyagari_params)
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid(p.w_min, p.w_max;
                                           length = p.N_w, spacing = :log)),
        StateAxis(:income, p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition = p.P_y)
    receipt = WealthChangeStage(layout;
        wealth_post = (cell; env) -> (1 + env.r) * cell.wealth + env.w * cell.income,
        wealth_axis = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end


# Production prices (plain function, no AbstractBlock) #
#------------------------------------------------------#

"Cobb-Douglas factor prices at aggregate capital `K`, fixed labor `p.L`."
function aiyagari_prices(K::Real, p = aiyagari_params)
    (; α, δ, L) = p
    r = α * (K / L)^(α - 1) - δ
    w = (1 - α) * (K / L)^α
    return (; r, w)
end
