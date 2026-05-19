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
# Permanent MIT shock: TFP steps from `A_ss = 1` to `A_0 > 1` at t = 1
# and stays there forever. The economy transitions to a new steady
# state at the higher TFP level.
#
# Algorithm sketch:
#   1. Solve the *initial* deterministic steady state at `A = 1` by
#      damped tatonnement on K. Pins down `K_ss^pre`, `V_ss^pre`,
#      `Λ_ss^pre`.
#   2. Solve the *terminal* deterministic steady state at `A = A_0` by
#      damped tatonnement on K. Pins down `K_ss^post`, `V_ss^post`.
#   3. Pick a path horizon `T`; choose a TFP path `{A_t}` = constant
#      `A_0`. Period-0 distribution is `Λ_ss^pre`; agents learn the
#      path at `t = 1` (perfect foresight).
#   4. Initialise `{K_t}` linearly interpolating from `K_ss^pre` →
#      `K_ss^post`. Iterate:
#      a. Backward sweep with terminal `V_{T+1} = V_ss^post`.
#      b. Forward sweep with initial `Λ_1 = Λ_ss^pre`; re-seat kernels
#         per period; read off `K^supplied_t`.
#      c. Damped update `K_t ← (1-d) K_t + d K^supplied_t`.
#   5. Stop when `‖K^supplied - K‖_∞ < tol`.

include("model.jl")

using Printf

"""
Solve the deterministic steady state at a fixed TFP level by damped
tatonnement on aggregate K. Returns the converged steady state plus
the household chain and `buffers` workspace so the SSJ demo can re-seat
kernels at the steady-state env without re-allocating.
"""
function mit_steady_state(p = mit_shock_params;
                          A::Float64   = 1.0,
                          K_init       = 5.0,
                          update_speed = 0.05,
                          rtol         = 2e-2,
                          max_iter     = 500,
                          verbosity    = 0)
    hh      = aiyagari_household(p)
    buffers = allocate(hh)

    K = K_init
    iterations = 0
    K_err = Inf
    residual_history = Float64[]
    V, Λ = nothing, nothing

    while iterations < max_iter
        env = (; K, aiyagari_prices(K, A, p)...)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env, buffers) :
            solve_steady_state_given_env!(hh, env, buffers;
                                          V_init = V, Λ_init = Λ)
        (; V, Λ, vfi_iters, lambda_iters) = res

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

    (; r, w) = aiyagari_prices(K, A, p)
    return (; K, r, w, V, Λ, hh, buffers,
              iterations, converged, residual_history)
end

"""
Compute the deterministic perfect-foresight transition path under a
permanent TFP shock from `A_ss = 1` to `A_0 > 1` at t = 1. The
household chain is the same shape as in the steady state; only the
env varies period by period via `aiyagari_prices(K_t, A_t, p)`.

Returns `(; K_path, A_path, V_path, Λ_path, K_ss_pre, K_ss_post, ...)`
along with iteration metadata.

`tol = 1e-3` is on `‖K_supplied - K‖_∞` (absolute, not relative). The
default `damping = 0.2` converges geometrically until it hits the
hard-argmax `ConsumptionSavingsStage` discretization floor (~2.6e-3 on
the baseline calibration); damping ≥ 0.3 oscillates near the impact
period.
"""
function mit_shock_transition(;
        T::Int           = 100,
        A_0::Float64     = 1.05,
        damping::Float64 = 0.2,
        tol::Float64     = 1e-3,
        maxiter::Int     = 200,
        p                = mit_shock_params,
        verbose::Bool    = true)
    # 1. Initial steady state (A = 1)
    verbose && println("Computing pre-shock steady state (A = 1)…")
    ss_pre = mit_steady_state(p; A = 1.0, verbosity = 0)
    K_ss_pre, V_ss_pre, Λ_ss_pre = ss_pre.K, ss_pre.V, ss_pre.Λ
    verbose && @printf "  K_ss_pre = %.4f, r = %.4f, w = %.4f\n" K_ss_pre ss_pre.r ss_pre.w

    # 2. Terminal steady state (A = A_0)
    verbose && println("Computing post-shock steady state (A = $(A_0))…")
    ss_post = mit_steady_state(p; A = A_0, K_init = K_ss_pre, verbosity = 0)
    K_ss_post, V_ss_post, Λ_ss_post = ss_post.K, ss_post.V, ss_post.Λ
    verbose && @printf "  K_ss_post = %.4f, r = %.4f, w = %.4f\n" K_ss_post ss_post.r ss_post.w

    # 3. Fresh chain + workspace for the path solve
    hh      = aiyagari_household(p)
    buffers = allocate(hh)
    dims    = layout_size(first(hh.stages).input_layout)

    # V_path[T+1] = V_ss_post (terminal condition); V_path[1..T] filled by backward sweep
    # Λ_path[1]   = Λ_ss_pre  (initial  condition); Λ_path[2..T+1] filled by forward sweep
    V_path = [copy(V_ss_post) for _ in 1:(T + 1)]
    Λ_path = [zeros(Float64, dims...) for _ in 1:(T + 1)]
    Λ_path[1] .= Λ_ss_pre

    # 4. Exogenous TFP path (permanent step)
    A_path = tfp_path(T; A_0 = A_0)

    # 5. Initial K guess: linear interpolation between the two SSs.
    K_path = collect(range(K_ss_pre, K_ss_post; length = T))
    K_supplied = zeros(T)
    residual_history = Float64[]
    iterations = 0
    res = Inf

    while iterations < maxiter
        # 5a. Backward sweep V_{T+1} = V_ss_post → V_T → … → V_1
        for t in T:-1:1
            env = (; K = K_path[t], aiyagari_prices(K_path[t], A_path[t], p)...)
            V_path[t] .= backward!(hh, V_path[t + 1], env, buffers)
        end

        # 5b. Forward sweep Λ_1 = Λ_ss_pre → … → Λ_{T+1}; read K^supplied_t
        for t in 1:T
            env = (; K = K_path[t], aiyagari_prices(K_path[t], A_path[t], p)...)
            # Re-seat per-stage kernels at period-t env (backward sweep
            # walked through later periods); only then is Λ-push consistent.
            backward!(hh, V_path[t + 1], env, buffers)
            Λ_path[t + 1] .= forward!(hh, Λ_path[t], buffers)
            K_supplied[t] = compute_moments(hh, env).K_supplied
        end

        # 5c. Residual + damped update
        res = maximum(abs, K_supplied .- K_path)
        push!(residual_history, res)
        iterations += 1
        verbose && (iterations <= 5 || iterations % 5 == 0) &&
            @printf "  iter %d: ‖K_supplied - K‖∞ = %.6f\n" iterations res

        res <= tol && break

        K_path .= (1 - damping) .* K_path .+ damping .* K_supplied
    end

    converged = res <= tol
    converged || @warn "MIT-shock transition: failed to reach tol = $tol in $maxiter iterations (last res = $res)."

    return (; K_path, A_path,
              V_path, Λ_path,
              K_ss_pre, K_ss_post,
              V_ss_pre, V_ss_post,
              Λ_ss_pre, Λ_ss_post,
              iterations, converged,
              residual_history)
end


# Run when executed as a script #
#-------------------------------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Solving Aiyagari MIT shock transition (permanent shock)…")
    @time tr = mit_shock_transition(T = 100, A_0 = 1.05,
                                     damping = 0.2, tol = 1e-3,
                                     maxiter = 200, verbose = true)
    println(tr.converged ? "Converged in $(tr.iterations) outer iterations." :
                            "DID NOT CONVERGE in $(tr.iterations) outer iterations.")
    @printf "  K_ss_pre   = %.4f\n" tr.K_ss_pre
    @printf "  K_ss_post  = %.4f\n" tr.K_ss_post
    @printf "  K[1]   (impact)  = %.4f\n" tr.K_path[1]
    @printf "  K[5]             = %.4f\n" tr.K_path[5]
    @printf "  K[20]            = %.4f\n" tr.K_path[20]
    @printf "  K[50]            = %.4f\n" tr.K_path[50]
    @printf "  K[100] (≈end)    = %.4f\n" tr.K_path[100]
    @printf "  A[1] / A[50] / A[100] = %.4f / %.4f / %.4f\n" tr.A_path[1] tr.A_path[50] tr.A_path[100]
    println("  Residual history (last 5): ",
            round.(tr.residual_history[max(1,end-4):end]; digits = 6))
end
