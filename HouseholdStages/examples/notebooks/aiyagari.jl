####################################################################
# Aiyagari (1994) — Heterogeneous-Agent Steady State (notebook)    #
####################################################################
#
# Self-contained driver: depends only on `HouseholdStages`. No
# `include` of sibling example folders. The within-period problem
# decomposes into three stages, in time order:
#
#     IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavingsStage
#
# (the canonical L03 / L04 decomposition). Walkthrough is organised
# in four numbered sections so a reader can step layer by layer:
# parameters and state layout, household stage chain and workspace,
# single-K inner solve, outer tatonnement driver.

using HouseholdStages
using Printf


# 1. Parameters and State Layout #
#--------------------------------#

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

_u_crra(c, ::Val{1}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)

# Assemble the moment-lifted household block, with the wealth-axis log
# grid and three-stage chain inlined here.
function aiyagari_household(p::AiyagariParams)
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

function aiyagari_prices(K::Real, p::AiyagariParams)
    (; α, δ, L) = p
    r = α * (K / L)^(α - 1) - δ
    w = (1 - α) * (K / L)^α
    return (; r, w)
end

p = AiyagariParams()
@printf "Calibration: β = %.3f, σ = %.2f, α = %.2f, δ = %.2f\n" p.β p.σ p.α p.δ


# 2. Household Stage Chain and Workspace #
#----------------------------------------#

# `hh` is the moment-lifted chain `shock ∘ₛ receipt ∘ₛ savings`. The
# `K_supplied = ∫ wealth dΛ` moment is read off at the end.

hh      = aiyagari_household(p)
buffers = allocate(hh)
dims    = layout_size(first(hh.stages).input_layout)

@printf "Layout: wealth %d × income %d = %d cells\n" dims[1] dims[2] prod(dims)


# 3. Inner Solve at a Single K #
#------------------------------#

# Fix a trial K, compute Cobb-Douglas (r, w), iterate V backward to its
# fixed point and Λ forward to stationarity via the bundled
# `solve_steady_state_given_env!` helper, then read off K_supplied via
# `compute_moments`. With no warm-start, V_init defaults to zeros and
# Λ_init to the uniform distribution.

K_trial   = 5.0
env_trial = (; K = K_trial, aiyagari_prices(K_trial, p)...)
(; V, Λ, vfi_iters, lambda_iters) =
    solve_steady_state_given_env!(hh, env_trial, buffers)
moms_trial = compute_moments(hh, env_trial)

@printf "K_trial = %.4f → K_supplied = %.4f (residual = %+.4f)\n" K_trial moms_trial.K_supplied (moms_trial.K_supplied - K_trial)
@printf "  VFI: %d iters; Λ: %d iters; r = %.4f, w = %.4f\n" vfi_iters lambda_iters env_trial.r env_trial.w


# 4. Outer Tatonnement #
#----------------------#

# Damped tatonnement on K: at each pass, run the inner V/Λ fixed
# point at the current K and nudge K toward the implied supply by
# `update_speed * (K_supplied - K)`. The library leaves "close-the-
# model" outer loops to the consumer.

update_speed = 0.05
rtol         = 2e-2
max_iter     = 500
K            = 5.0
residual_history = Float64[]
K_err = Inf
iters = 0

println("Solving Aiyagari steady state…")
@time begin
    while iters < max_iter
        env = (; K, aiyagari_prices(K, p)...)
        res = solve_steady_state_given_env!(hh, env, buffers; V_init = V, Λ_init = Λ)
        global V, Λ = res.V, res.Λ

        K_supplied = compute_moments(hh, env).K_supplied
        global K_err = abs(K_supplied - K) / K
        push!(residual_history, K_err)
        global iters += 1

        @printf "  iter %d: K = %.4f → K_supplied = %.4f, K_err = %.6f, VFI %d / Λ %d\n" iters K K_supplied K_err res.vfi_iters res.lambda_iters

        K_err <= rtol && break

        global K += update_speed * (K_supplied - K)
    end
end

converged = K_err <= rtol
println(converged ? "Converged in $iters outer iterations." :
                    "DID NOT CONVERGE in $iters outer iterations.")
(; r, w) = aiyagari_prices(K, p)
@printf "  K   = %.4f\n" K
@printf "  r   = %.4f\n" r
@printf "  w   = %.4f\n" w
@printf "  ΣΛ  = %.6f\n" sum(Λ)
println("  Residual history (last 5): $(round.(residual_history[max(1,end-4):end]; digits = 6))")
