####################################################################
# Spatial Two-Location Steady State (notebook)                     #
####################################################################
#
# Self-contained driver: depends only on `HouseholdStages`. Adds a
# `:location` axis (categorical over `[:home, :abroad]`) to the
# Aiyagari chain and a dedicated `MigrationStage` between the income
# shock and the L03/L04 savings decomposition:
#
#     IncomeShock ∘ MigrationStage ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# Each location's capital market clears independently — no inter-
# location trade. Outer loop: damped tatonnement on
# `(K_home, K_abroad)`.

using HouseholdStages
using Printf


# 1. Parameters and State Layout #
#--------------------------------#

@kwdef struct SpatialParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36
    δ :: Float64       = 0.08

    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]

    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 30.0

    L :: Float64       = 1.0
    A_home::Float64    = 1.0
    A_abroad::Float64  = 1.0

    ε_logit::Float64        = 5.0
    migration_cost::Float64 = 0.5
end
Base.Broadcast.broadcastable(p::SpatialParams) = Ref(p)

_u_crra(c, ::Val{1}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)

function spatial_household(p::SpatialParams)
    layout = StateLayout(
        StateAxis(:wealth,   continuous_grid(p.w_min, p.w_max;
                                             length = p.N_w, spacing = :log)),
        StateAxis(:income,   p.y_grid),
        StateAxis(:location, categorical([:home, :abroad])),
    )

    shock = MarkovStage(layout; axis = :income, transition = p.P_y)

    migration = MigrationStage(layout;
        location_axis  = :location,
        migration_cost = [0.0              p.migration_cost;
                          p.migration_cost 0.0],
        ε              = p.ε_logit,
    )

    receipt = WealthChangeStage(layout;
        wealth_post = function (cell; env)
            (r, w) = cell.location == :home ?
                     (env.r_home, env.w_home) :
                     (env.r_abroad, env.w_abroad)
            return (1 + r) * cell.wealth + w * cell.income
        end,
        wealth_axis = :wealth,
    )

    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )

    return define_moments!(shock ∘ migration ∘ receipt ∘ savings;
        K_home     = at_end(integrand = (cell; env) -> cell.location == :home   ? cell.wealth : 0.0,
                            reduce = sum),
        K_abroad   = at_end(integrand = (cell; env) -> cell.location == :abroad ? cell.wealth : 0.0,
                            reduce = sum),
        pop_home   = at_end(integrand = (cell; env) -> cell.location == :home   ? 1.0 : 0.0,
                            reduce = sum),
        pop_abroad = at_end(integrand = (cell; env) -> cell.location == :abroad ? 1.0 : 0.0,
                            reduce = sum),
    )
end

function spatial_prices(K_home::Real, K_abroad::Real, p::SpatialParams)
    (; α, δ, L, A_home, A_abroad) = p
    L_each   = L / 2
    r_home   = α * A_home   * (K_home   / L_each)^(α - 1) - δ
    w_home   = (1 - α) * A_home   * (K_home   / L_each)^α
    r_abroad = α * A_abroad * (K_abroad / L_each)^(α - 1) - δ
    w_abroad = (1 - α) * A_abroad * (K_abroad / L_each)^α
    return (; r_home, w_home, r_abroad, w_abroad)
end

p    = SpatialParams()
@printf "Calibration: β = %.3f, σ = %.2f, α = %.2f, δ = %.2f\n" p.β p.σ p.α p.δ
@printf "Migration: ε_logit = %.2f, migration_cost = %.2f\n" p.ε_logit p.migration_cost


# 2. Household Stage Chain #
#--------------------------#

hh   = spatial_household(p)
dims = layout_size(first(hh.spec.stages).input_layout)

@printf "Layout: wealth %d × income %d × location %d = %d cells\n" dims[1] dims[2] dims[3] prod(dims)


# 3. Inner Solve at a Trial (K_home, K_abroad) #
#----------------------------------------------#

K_home_trial   = 3.0
K_abroad_trial = 3.0
pr_trial = spatial_prices(K_home_trial, K_abroad_trial, p)
env_trial = (; K_home = K_home_trial, K_abroad = K_abroad_trial,
               pr_trial.r_home, pr_trial.w_home,
               pr_trial.r_abroad, pr_trial.w_abroad)
res_trial = solve_steady_state_given_env!(hh, env_trial)
V, Λ      = res_trial.V, res_trial.Λ
m_trial   = compute_moments(hh, Λ, env_trial)

@printf "(K_h, K_a)_trial = (%.4f, %.4f) → (Ks_h, Ks_a) = (%.4f, %.4f)\n" K_home_trial K_abroad_trial m_trial.K_home m_trial.K_abroad
@printf "  residuals = (%+.4f, %+.4f); pop_home = %.4f, pop_abroad = %.4f\n" (m_trial.K_home - K_home_trial) (m_trial.K_abroad - K_abroad_trial) m_trial.pop_home m_trial.pop_abroad
@printf "  VFI: %d iters; Λ: %d iters; r_h / r_a = %.4f / %.4f\n" res_trial.history.vfi_iters res_trial.history.lambda_iters env_trial.r_home env_trial.r_abroad


# 4. Outer Damped Tatonnement #
#-----------------------------#

damping = 0.1
tol     = 0.25
maxiter = 120

K_home   = 3.0
K_abroad = 3.0
res_history = Tuple{Float64, Float64}[]
moments_last = nothing
iters = 0
converged = false

println("Solving spatial two-location steady state…")
@time begin
    while iters < maxiter
        pr  = spatial_prices(K_home, K_abroad, p)
        env = (; K_home, K_abroad, pr.r_home, pr.w_home, pr.r_abroad, pr.w_abroad)
        res = solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        global V, Λ = res.V, res.Λ
        global moments_last = compute_moments(hh, Λ, env)

        res_h = moments_last.K_home   - K_home
        res_a = moments_last.K_abroad - K_abroad
        push!(res_history, (res_h, res_a))
        global iters += 1

        @printf "  iter %d: K_h = %.3f, K_a = %.3f → Ks_h = %.3f, Ks_a = %.3f; res = (%+.4f, %+.4f)\n" iters K_home K_abroad moments_last.K_home moments_last.K_abroad res_h res_a

        if max(abs(res_h), abs(res_a)) < tol
            global converged = true
            break
        end

        global K_home   = (1 - damping) * K_home   + damping * moments_last.K_home
        global K_abroad = (1 - damping) * K_abroad + damping * moments_last.K_abroad
    end
end

println(converged ? "Converged in $iters outer iterations." :
                    "DID NOT CONVERGE in $iters outer iterations.")
prices_final = spatial_prices(K_home, K_abroad, p)
@printf "  K_home     = %.4f\n" K_home
@printf "  K_abroad   = %.4f\n" K_abroad
@printf "  pop_home   = %.4f\n" moments_last.pop_home
@printf "  pop_abroad = %.4f\n" moments_last.pop_abroad
@printf "  r_home / r_abroad = %.4f / %.4f\n" prices_final.r_home prices_final.r_abroad
@printf "  w_home / w_abroad = %.4f / %.4f\n" prices_final.w_home prices_final.w_abroad
@printf "  ΣΛ                = %.6f\n" sum(Λ)
println("  Residual history (last 3): ",
        [(round(r_h; digits = 4), round(r_a; digits = 4))
         for (r_h, r_a) in res_history[max(1, end - 2):end]])
