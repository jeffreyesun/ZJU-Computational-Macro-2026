############################################
# Aiyagari Steady State — Tatonnement on K #
############################################

# Damped tatonnement on aggregate capital K. The per-K inner work
# (V backward to fixed point + Λ forward to stationarity) is delegated
# to `HouseholdStages.solve_steady_state_given_env!`; the outer
# tatonnement loop is rolled here — the library deliberately leaves
# the "close-the-model" outer loop to consumers (calibration choices,
# update rules, residual semantics are example-specific). See
# `PROJECT_PLAN.md` Decisions log 2026-05-18 for the principle.
#
# The full walkthrough lives in `../notebooks/aiyagari.jl`.

include("model.jl")

using Printf

"""
Solve the Aiyagari steady state by damped tatonnement on aggregate K:
at each pass, run the inner V/Λ solve at the current K and nudge K
toward the implied supply by `update_speed * (K_supplied - K)`.

Hard-argmax `ConsumptionSavings` makes `K_supplied(K)` a step function
in K, so the residual has a discretization floor (≈ 1–2% at N_w = 400);
the default `rtol = 0.02` accepts that floor.
"""
function aiyagari_steady_state(p = aiyagari_params;
                               K_init       = 5.0,
                               update_speed = 0.05,
                               rtol         = 2e-2,
                               max_iter     = 500,
                               verbosity    = 1)
    hh   = aiyagari_household(p)
    caches, scratches = allocate(hh)
    dims = layout_size(aiyagari_layout(p))
    V    = zeros(Float64, dims...)
    Λ    = fill(1.0 / prod(dims), dims...)

    K = K_init
    iterations = 0
    K_err = Inf
    residual_history = Float64[]
    info = nothing

    while iterations < max_iter
        env  = (; K, aiyagari_prices(K, p)...)
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
    converged || @warn "Aiyagari tatonnement stuck at tolerance $rtol after $iterations iterations. Returning current K."

    (; r, w) = aiyagari_prices(K, p)
    return (; K, r, w, V, Λ,
              iterations, converged,
              residual_history)
end


# Driver #
#--------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Aiyagari deterministic steady state…")
    @time res = aiyagari_steady_state()
    println(res.converged ? "Converged in $(res.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(res.iterations) outer iterations.")
    @printf "  K   = %.4f\n" res.K
    @printf "  r   = %.4f\n" res.r
    @printf "  w   = %.4f\n" res.w
    @printf "  ΣΛ  = %.6f\n" sum(res.Λ)
end
