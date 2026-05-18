###################################################
# Krusell-Smith Deterministic Steady State Driver #
###################################################

# Damped tatonnement on aggregate K at constant TFP `A = 1.0`. The
# per-K inner work is delegated to
# `HouseholdStages.solve_steady_state_given_env!`; the outer tatonnement
# loop is rolled here — the library leaves "close-the-model" outer
# loops to consumers (see `PROJECT_PLAN.md` Decisions log 2026-05-18).
#
# Bisection isn't applicable here: bisection's extreme-K probes push
# r as high as 0.30, and `WealthChange.backward`'s linear V
# extrapolation amplifies V faster than `1/β` at those rates, breaking
# the Bellman contraction. Tatonnement stays near the candidate K
# throughout, never visiting that high-r regime.

include("model.jl")

using Printf

"""
Solve the K-S deterministic steady state by damped tatonnement on K
at constant TFP `A`. Hard-argmax `ConsumptionSavings` produces a step
function in K; K-S sits at `β(1+r) ≈ 1`, so a one-grid-step policy
switch moves `K_supplied` by ~20%. Default `rtol = 0.05` accepts the
floor and `update_speed = 0.01` damps heavily to walk through it.
"""
function ks_steady_state(p = ks_params;
                         A::Float64       = 1.0,
                         K_init           = 13.6,
                         update_speed     = 0.01,
                         rtol             = 5e-2,
                         max_iter         = 500,
                         verbosity        = 1)
    hh   = ks_household(p)
    caches, scratches = allocate(hh)
    dims = layout_size(ks_layout(p))
    V    = zeros(Float64, dims...)
    Λ    = fill(1.0 / prod(dims), dims...)

    K = K_init
    iterations = 0
    K_err = Inf
    residual_history = Float64[]

    while iterations < max_iter
        env  = (; K, A, ks_prices(K, A, p)...)
        info = solve_steady_state_given_env!(hh, env, V, Λ, caches, scratches)
        V, Λ = info.V, info.Λ

        K_supplied = compute_moments(hh, env).K_supplied
        K_err = abs(K_supplied - K) / K
        push!(residual_history, K_err)
        iterations += 1

        verbosity > 0 && @printf "  iter %d: K = %.4f → K_supplied = %.4f, K_err = %.6f, VFI iters = %d, Λ iters = %d\n" iterations K K_supplied K_err info.vfi_iters info.lambda_iters

        K_err <= rtol && break

        K += update_speed * (K_supplied - K)
    end

    converged = K_err <= rtol
    converged || @warn "K-S tatonnement stuck at tolerance $rtol after $iterations iterations. Returning current K."

    (; r, w) = ks_prices(K, A, p)
    return (; K, r, w, V, Λ,
              iterations, converged,
              residual_history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving K-S deterministic steady state (A = 1.0, β = 0.96)…")
    @time res = ks_steady_state()
    println(res.converged ? "Converged in $(res.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(res.iterations) outer iterations.")
    @printf "  K   = %.4f\n" res.K
    @printf "  r   = %.4f\n" res.r
    @printf "  w   = %.4f\n" res.w
    @printf "  ΣΛ  = %.6f\n" sum(res.Λ)
end
