###############################################################
# Aiyagari MIT Shock — Perfect-Foresight Transition Driver    #
###############################################################

# The per-env inner work (V backward and Λ forward fixed points at a
# single env) comes from `HouseholdStages.solve_steady_state_given_env!`.
# The outer tatonnement loop — both for the steady-state warm start
# and for the transition path — is rolled here, because the library
# deliberately leaves "close-the-model" outer loops to consumers
# (see `PROJECT_PLAN.md` Decisions log 2026-05-18).
#
# Algorithm sketch:
#   1. Solve the deterministic steady state at `A = 1` by damped
#      tatonnement on K. Pins down `K_ss`, `V_ss`, `Λ_ss`.
#   2. Choose a TFP path `{A_t}` decaying from `A_0 > 1` back to `A_ss`.
#      Period-0 distribution is `Λ_ss`; agents learn the entire path
#      at `t = 1` (perfect foresight).
#   3. Initialise `{K_t}` at `K_ss`. Iterate:
#      a. Backward sweep with terminal `V_{T+1} = V_ss`: for
#         `t = T, …, 1`, V_t = backward!(hh, V_{t+1}, env_t).
#      b. Forward sweep with initial `Λ_1 = Λ_ss`: for `t = 1, …, T`,
#         re-seat caches at period-t env (the backward sweep walked
#         through later periods), forward Λ, read off K^supplied_t.
#      c. Damped update `K_t ← (1-d) K_t + d K^supplied_t`.
#   4. Stop when `‖K^supplied - K‖_∞ < tol`.

include("model.jl")

using Printf

"""
Solve the deterministic steady state (`A_ss = 1`) by damped tatonnement
on aggregate K. Returns the converged steady state plus the household
chain and its workspace (caches, scratches) so the SSJ demo can re-seat
them at the steady-state env without re-allocating.
"""
function mit_steady_state(p = mit_shock_params;
                          K_init       = 5.0,
                          update_speed = 0.05,
                          rtol         = 2e-2,
                          max_iter     = 500,
                          verbosity    = 0)
    hh   = aiyagari_household(p)
    caches, scratches = allocate(hh)
    dims = layout_size(aiyagari_layout(p))
    V    = zeros(Float64, dims...)
    Λ    = fill(1.0 / prod(dims), dims...)

    K = K_init
    iterations = 0
    K_err = Inf
    residual_history = Float64[]

    while iterations < max_iter
        env  = (; K, aiyagari_prices(K, 1.0, p)...)
        info = solve_steady_state_given_env!(hh, env, V, Λ, caches, scratches)
        V, Λ = info.V, info.Λ

        K_supplied = compute_moments(hh, env).K_supplied
        K_err = abs(K_supplied - K) / K
        push!(residual_history, K_err)
        iterations += 1

        verbosity > 0 && @printf "  SS iter %d: K = %.4f → K_supplied = %.4f, K_err = %.6f\n" iterations K K_supplied K_err

        K_err <= rtol && break

        K += update_speed * (K_supplied - K)
    end

    converged = K_err <= rtol
    converged || @warn "MIT-shock SS tatonnement stuck at tolerance $rtol after $iterations iterations."

    (; r, w) = aiyagari_prices(K, 1.0, p)
    return (; K, r, w, V, Λ, hh, caches, scratches,
              iterations, converged, residual_history)
end

"""
Compute the deterministic perfect-foresight transition path under a
one-time unanticipated TFP shock with AR(1) decay. The household
chain is the same shape as in the steady state; only the env varies
period by period via `aiyagari_prices(K_t, A_t, p)`.

Returns the converged `(K_path, A_path, V_path, Λ_path)` along with
the steady-state warm start `K_ss` and iteration metadata.

`tol = 1e-3` is on `‖K_supplied - K‖_∞` (absolute, not relative). The
default `damping = 0.2` converges geometrically until it hits the
hard-argmax `ConsumptionSavings` discretization floor (~2.6e-3 on the
baseline calibration); damping ≥ 0.3 oscillates near the impact period.
"""
function mit_shock_transition(;
        T::Int           = 100,
        A_0::Float64     = 1.05,
        ρ::Float64       = 0.85,
        damping::Float64 = 0.2,
        tol::Float64     = 1e-3,
        maxiter::Int     = 200,
        p                = mit_shock_params,
        verbose::Bool    = true)
    # 1. Steady state (warm start)
    verbose && println("Computing steady state…")
    ss = mit_steady_state(p; verbosity = 0)
    K_ss, V_ss, Λ_ss = ss.K, ss.V, ss.Λ
    verbose && @printf "  K_ss = %.4f, r_ss = %.4f, w_ss = %.4f\n" K_ss ss.r ss.w

    # 2. Build a fresh chain + workspace for the path solve
    hh   = aiyagari_household(p)
    caches, scratches = allocate(hh)
    dims = layout_size(aiyagari_layout(p))

    # V_path[T+1] = V_ss (terminal condition); V_path[1..T] filled by backward sweep
    # Λ_path[1]   = Λ_ss (initial  condition); Λ_path[2..T+1] filled by forward sweep
    V_path = [copy(V_ss) for _ in 1:(T + 1)]
    Λ_path = [zeros(Float64, dims...) for _ in 1:(T + 1)]
    Λ_path[1] .= Λ_ss

    # 3. Exogenous TFP path
    A_path = tfp_path(T; A_0 = A_0, ρ = ρ, A_ss = 1.0)

    # 4. Initial K guess: hold at SS
    K_t = fill(K_ss, T)
    K_supplied = zeros(T)
    residual_history = Float64[]
    iterations = 0
    res = Inf

    while iterations < maxiter
        # 4a. Backward sweep V_{T+1} = V_ss → V_T → … → V_1
        for t in T:-1:1
            env = (; K = K_t[t], aiyagari_prices(K_t[t], A_path[t], p)...)
            V_path[t] .= backward!(hh, V_path[t + 1], env, caches, scratches)
        end

        # 4b. Forward sweep Λ_1 = Λ_ss → … → Λ_{T+1}; read K^supplied_t
        for t in 1:T
            env = (; K = K_t[t], aiyagari_prices(K_t[t], A_path[t], p)...)
            # Re-seat per-stage caches at period-t env (backward sweep
            # walked through later periods); only then is Λ-push consistent.
            backward!(hh, V_path[t + 1], env, caches, scratches)
            Λ_path[t + 1] .= forward!(hh, Λ_path[t], caches, scratches)
            K_supplied[t] = compute_moments(hh, env).K_supplied
        end

        # 4c. Residual + damped update
        res = maximum(abs, K_supplied .- K_t)
        push!(residual_history, res)
        iterations += 1
        verbose && (iterations <= 5 || iterations % 5 == 0) &&
            @printf "  iter %d: ‖K_supplied - K‖∞ = %.6f\n" iterations res

        res <= tol && break

        K_t .= (1 - damping) .* K_t .+ damping .* K_supplied
    end

    converged = res <= tol
    converged || @warn "MIT-shock transition: failed to reach tol = $tol in $maxiter iterations (last res = $res)."

    return (; K_path = K_t, A_path,
              V_path, Λ_path,
              K_ss, V_ss, Λ_ss,
              iterations, converged,
              residual_history)
end


# Run when executed as a script #
#-------------------------------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Aiyagari MIT shock transition…")
    @time tr = mit_shock_transition(T = 100, A_0 = 1.05, ρ = 0.85,
                                     damping = 0.2, tol = 1e-3,
                                     maxiter = 200, verbose = true)
    println(tr.converged ? "Converged in $(tr.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(tr.iterations) outer iterations.")
    @printf "  K_ss             = %.4f\n" tr.K_ss
    @printf "  K[1]   (impact)  = %.4f\n" tr.K_path[1]
    @printf "  K[5]             = %.4f\n" tr.K_path[5]
    @printf "  K[20]            = %.4f\n" tr.K_path[20]
    @printf "  K[50]            = %.4f\n" tr.K_path[50]
    @printf "  K[100] (≈end)    = %.4f\n" tr.K_path[100]
    @printf "  A[1] / A[10] / A[50] = %.4f / %.4f / %.4f\n" tr.A_path[1] tr.A_path[10] tr.A_path[50]
    println("  Residual history (last 5): ",
            round.(tr.residual_history[max(1,end-4):end]; digits = 6))
end
