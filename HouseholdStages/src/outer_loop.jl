###############################################################
# Inner-solve helpers at a fixed env                          #
###############################################################
#
# The only "outer-loop" code that belongs in `HouseholdStages/src/` is
# the code that takes `env` for granted and operates only on the
# household chain: backward-iterate V to its fixed point, forward-
# iterate Λ to stationarity, and the bundle of the two. Anything that
# closes the model around the household block — tatonnement on K, the
# MIT-shock transition path, the AR(1) TFP path — is calibration-
# specific and lives in the consuming example or notebook.
#
# The 2026-05-18 first-pass refactor briefly promoted a generic
# `solve_picard_steady_state` / `solve_picard_transition` /
# `tfp_path`; the user reversed that on the same day on the grounds
# that those helpers DON'T satisfy the "given env, on the chain"
# discipline. See `PROJECT_PLAN.md` Decisions log.

using Printf


# Inner V fixed point #
#---------------------#

"""
    solve_vfi_steady_state_given_env!(hh, env, V0, caches, scratches;
                                       tol=1e-7, maxiter=1500)
        -> (V, iters, converged)

Backward-iterate `V` to its fixed point at `env` by repeatedly applying
`backward!` on the household chain `hh`. Returns the converged value
function, the number of iterations taken, and a `converged` flag. The
caller owns `V0` and the `(caches, scratches)` workspace produced by
`allocate(hh)`.

Throws an error if `maxiter` is reached before `‖V_new − V‖_∞ ≤ tol`.
"""
function solve_vfi_steady_state_given_env!(hh, env, V0, caches, scratches;
                                            tol::Real = 1e-7,
                                            maxiter::Int = 1500)
    V = copy(V0)
    diff = Inf
    iters = 0
    while diff > tol
        V_new = backward!(hh, V, env, caches, scratches)
        diff  = maximum(abs, V_new .- V)
        V .= V_new
        iters += 1
        iters == maxiter && error("solve_vfi_steady_state_given_env!: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (V = V, iters = iters, converged = true)
end


# Inner Λ fixed point #
#---------------------#

"""
    solve_lambda_steady_state_given_env!(hh, Λ0, caches, scratches;
                                          tol=1e-6, maxiter=20_000)
        -> (Λ, iters, converged)

Forward-iterate `Λ` to its stationary distribution by repeatedly
applying `forward!` on the household chain `hh`. The policy in `caches`
is whatever the most recent `backward!` call populated; the caller is
responsible for having seeded that at the env this iteration is
intended for.

Throws an error if `maxiter` is reached before `‖Λ_new − Λ‖_∞ ≤ tol`.
"""
function solve_lambda_steady_state_given_env!(hh, Λ0, caches, scratches;
                                               tol::Real = 1e-6,
                                               maxiter::Int = 20_000)
    Λ = copy(Λ0)
    diff = Inf
    iters = 0
    while diff > tol
        Λ_new = forward!(hh, Λ, caches, scratches)
        diff  = maximum(abs, Λ_new .- Λ)
        Λ .= Λ_new
        iters += 1
        iters == maxiter && error("solve_lambda_steady_state_given_env!: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (Λ = Λ, iters = iters, converged = true)
end


# Bundle: V fixed point, then Λ fixed point, at a single env #
#-----------------------------------------------------------#

"""
    solve_steady_state_given_env!(hh, env, V0, Λ0, caches, scratches;
                                   vfi_tol=1e-7, vfi_maxiter=1500,
                                   lambda_tol=1e-6, lambda_maxiter=20_000)
        -> (V, Λ, vfi_iters, lambda_iters)

Bundle of the two single-env inner solves: run `V` to its fixed point
at `env`, then `Λ` to stationarity using the policy that backward
populated in `caches`. Returns the converged `(V, Λ)` along with the
inner iteration counts.

This helper takes `env` for granted — it does NOT touch any outer
loop, does NOT compute moments, and does NOT update an aggregate
state. Wrap it in a tatonnement loop in the consumer (per-example
driver / notebook) and call `compute_moments(hh, env)` after for
moment readout.
"""
function solve_steady_state_given_env!(hh, env, V0, Λ0, caches, scratches;
                                        vfi_tol::Real    = 1e-7,
                                        vfi_maxiter::Int = 1500,
                                        lambda_tol::Real = 1e-6,
                                        lambda_maxiter::Int = 20_000)
    vfi = solve_vfi_steady_state_given_env!(hh, env, V0, caches, scratches;
                                            tol = vfi_tol, maxiter = vfi_maxiter)
    lam = solve_lambda_steady_state_given_env!(hh, Λ0, caches, scratches;
                                                tol = lambda_tol, maxiter = lambda_maxiter)
    return (V = vfi.V, Λ = lam.Λ,
            vfi_iters = vfi.iters, lambda_iters = lam.iters)
end
