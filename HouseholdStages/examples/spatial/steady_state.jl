######################################
# Spatial Two-Location Steady State #
######################################

# Damped tatonnement on `(K_home, K_abroad)` under location-specific
# TFP. The per-pair inner work is delegated to
# `HouseholdStages.solve_steady_state_given_env!`; the outer
# tatonnement is rolled here (see `PROJECT_PLAN.md` Decisions log
# 2026-05-18 for the principle).

include("model.jl")

using Printf

"""
Solve the spatial two-location steady state by damped tatonnement on
the K-pair. The tolerance `tol = 0.25` (≈8% of per-location K) accepts
the hard-argmax `ConsumptionSavingsStage` step-function floor.
"""
function spatial_steady_state(; p = params,
                                K_home_0::Float64   = 3.0,
                                K_abroad_0::Float64 = 3.0,
                                damping::Float64    = 0.1,
                                tol::Float64        = 0.25,
                                maxiter::Int        = 120,
                                verbosity::Int      = 1)
    hh = spatial_household(p)

    K_home, K_abroad = K_home_0, K_abroad_0
    iterations = 0
    converged  = false
    res_history = Tuple{Float64, Float64}[]
    moments = nothing
    env     = nothing
    V, Λ    = nothing, nothing   # warm-start placeholders

    while iterations < maxiter
        pr  = spatial_prices(K_home, K_abroad, p)
        env = (; K_home, K_abroad, pr.r_home, pr.w_home, pr.r_abroad, pr.w_abroad)

        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env;
                                          V_init = V, Λ_init = Λ)
        (; V, Λ, history) = res; (; vfi_iters, lambda_iters) = history
        moments = compute_moments(hh, Λ, env)

        res_h = moments.K_home   - K_home
        res_a = moments.K_abroad - K_abroad
        push!(res_history, (res_h, res_a))
        iterations += 1

        verbosity > 0 && (iterations % 5 == 1 || iterations <= 6) && @printf(
            "  iter %d: K_h = %.3f, K_a = %.3f → Ks_h = %.3f, Ks_a = %.3f; res = (%+.4f, %+.4f); VFI %d / Λ %d\n",
            iterations, K_home, K_abroad,
            moments.K_home, moments.K_abroad, res_h, res_a,
            vfi_iters, lambda_iters)

        if max(abs(res_h), abs(res_a)) < tol
            converged = true
            break
        end

        K_home   = (1 - damping) * K_home   + damping * moments.K_home
        K_abroad = (1 - damping) * K_abroad + damping * moments.K_abroad
    end

    converged || @warn "Spatial tatonnement stuck at tolerance $tol after $iterations iterations. Returning current K pair."

    return (; K_home, K_abroad,
              pop_home   = moments.pop_home,
              pop_abroad = moments.pop_abroad,
              V, Λ, env,
              iterations, converged,
              residual_history = res_history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving spatial two-location steady state…")
    @time res = spatial_steady_state()
    println(res.converged ? "Converged in $(res.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(res.iterations) outer iterations.")
    @printf "  K_home     = %.4f\n" res.K_home
    @printf "  K_abroad   = %.4f\n" res.K_abroad
    @printf "  pop_home   = %.4f\n" res.pop_home
    @printf "  pop_abroad = %.4f\n" res.pop_abroad
    @printf "  r_home / r_abroad = %.4f / %.4f\n" res.env.r_home res.env.r_abroad
    @printf "  w_home / w_abroad = %.4f / %.4f\n" res.env.w_home res.env.w_abroad
    @printf "  ΣΛ                = %.6f\n" sum(res.Λ)
end
