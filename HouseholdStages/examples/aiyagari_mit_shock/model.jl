#####################################################
# Aiyagari with MIT Shock — Model Definition         #
#####################################################

# Same within-period structure as Aiyagari (`../aiyagari/model.jl`),
# specialised to a transition path under a one-time unanticipated TFP
# shock. The household problem decomposes into three stages, in time
# order:
#
#     IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavingsStage
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
# imported from `../aiyagari/model.jl`. The wealth grid is log-spaced
# for the same reason as in Aiyagari.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct MITShockParams
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

Base.Broadcast.broadcastable(p::MITShockParams) = Ref(p)

const mit_shock_params = MITShockParams()


# Utility #
#---------#

_u_crra(c, ::Union{Val{1}, Val{1.0}}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)


# Household chain assembly #
#--------------------------#

"""
Build the moment-lifted Aiyagari household block
`IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings` (same shape as
the steady-state example).
"""
function aiyagari_household(p = mit_shock_params)
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

    return lift_moments(shock ∘ₛ receipt ∘ₛ savings;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
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
    return (; r, w)
end


# TFP shock path — permanent step #
#---------------------------------#

"""CLAUDE
Permanent TFP shock: `A[t] = A_0` for all `t = 1..T`. The new TFP level
arrives at period 1 (unanticipated at t = 0, then anticipated forever
after) and stays there. Returns a length-`T` vector. Defined locally
because the library deliberately leaves transition-side utilities to
the consumer (see `PROJECT_PLAN.md` Decisions log 2026-05-18).
"""
function tfp_path(T::Int; A_0::Float64 = 1.05)
    return fill(A_0, T)
end
