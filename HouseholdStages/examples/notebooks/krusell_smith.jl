####################################################################
# Krusell-Smith (1998) — Deterministic-Aggregate Steady State      #
####################################################################
#
# Self-contained driver: depends only on `HouseholdStages`. Same
# within-period structure as Aiyagari, specialised to the K-S
# employed/unemployed income process:
#
#     IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavingsStage
#
# Calibration: β = 0.96 (annual-style), log utility (γ = 1), two-state
# income with `y_unemp = 0.07` (canonical K-S — a strictly positive
# unemployed income is required under log utility so the b = 0 corner
# stays feasible). 400-point log-spaced wealth grid on [0, 200], wider
# /finer than Aiyagari's because K-S sits at the impatience watershed
# `β(1+r) ≈ 1`.

using HouseholdStages
using LinearAlgebra: I
using Printf


# 1. Parameters and State Layout #
#--------------------------------#

@kwdef struct KSParams
    β :: Float64       = 0.96
    γ :: Float64       = 1.0
    α :: Float64       = 0.36
    δ :: Float64       = 0.025
    y_grid :: Vector{Float64} = [0.07, 1.0]
    P_y    :: Matrix{Float64} = [0.6   0.4;
                                 0.05  0.95]
    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 200.0
end
Base.Broadcast.broadcastable(p::KSParams) = Ref(p)

_u_crra(c, ::Union{Val{1}, Val{1.0}}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)

function ks_effective_labor(P_y::AbstractMatrix, y_grid::AbstractVector)
    n = size(P_y, 1)
    A = P_y' - I(n)
    A[end, :] .= 1.0
    rhs = zeros(n); rhs[end] = 1.0
    π = A \ rhs
    return sum(y_grid .* π)
end

function ks_household(p::KSParams)
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
        utility         = (cell, c; env) -> u_crra(c, Val(p.γ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
    return lift_moments(shock ∘ₛ receipt ∘ₛ savings;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end

function ks_prices(K::Real, A::Real, p::KSParams)
    (; α, δ) = p
    L = ks_effective_labor(p.P_y, p.y_grid)
    r = α * A * (K / L)^(α - 1) - δ
    w = (1 - α) * A * (K / L)^α
    return (; r, w)
end

p     = KSParams()
L_eff = ks_effective_labor(p.P_y, p.y_grid)
@printf "Calibration: β = %.3f, γ = %.2f, α = %.2f, δ = %.3f\n" p.β p.γ p.α p.δ
@printf "Income process: y_unemp = %.2f, y_emp = %.2f, L_eff = %.4f\n" p.y_grid[1] p.y_grid[2] L_eff


# 2. Household Stage Chain and Workspace #
#----------------------------------------#

hh      = ks_household(p)
buffers = allocate(hh)
dims    = layout_size(first(hh.stages).input_layout)

@printf "Layout: wealth %d × income %d = %d cells\n" dims[1] dims[2] prod(dims)


# 3. Inner Solve at a Single K #
#------------------------------#

# K-S sits at `β(1+r) ≈ 1`, so K_supplied(K) is highly nonlinear near
# the equilibrium: a small change in K can flip the argmax policy and
# shift K_supplied by ~20%. The outer driver in section 4 damps
# heavily to walk through this noisy region.

K_trial   = 13.6
env_trial = (; K = K_trial, A = 1.0, ks_prices(K_trial, 1.0, p)...)
(; V, Λ, vfi_iters, lambda_iters) =
    solve_steady_state_given_env!(hh, env_trial, buffers)
moms_trial = compute_moments(hh, env_trial)

@printf "K_trial = %.4f → K_supplied = %.4f (residual = %+.4f)\n" K_trial moms_trial.K_supplied (moms_trial.K_supplied - K_trial)
@printf "  VFI: %d iters; Λ: %d iters; r = %.4f, w = %.4f, β(1+r) = %.5f\n" vfi_iters lambda_iters env_trial.r env_trial.w p.β * (1 + env_trial.r)


# 4. Outer Tatonnement #
#----------------------#

update_speed = 0.01
rtol         = 5e-2
max_iter     = 500
K            = 13.6
residual_history = Float64[]
K_err = Inf
iters = 0

println("Solving Krusell-Smith deterministic steady state…")
@time begin
    while iters < max_iter
        env = (; K, A = 1.0, ks_prices(K, 1.0, p)...)
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
(; r, w) = ks_prices(K, 1.0, p)
@printf "  K   = %.4f\n" K
@printf "  r   = %.4f\n" r
@printf "  w   = %.4f\n" w
@printf "  β(1+r) = %.5f\n" p.β * (1 + r)
@printf "  ΣΛ  = %.6f\n" sum(Λ)
println("  Residual history (last 5): $(round.(residual_history[max(1,end-4):end]; digits = 6))")
