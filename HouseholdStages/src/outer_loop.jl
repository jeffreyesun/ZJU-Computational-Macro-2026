###############################################################
# Inner-solve helpers at a fixed env                          #
###############################################################
#
# The only "outer-loop" code that belongs in `HouseholdStages/src/` is
# the code that takes `env` for granted and operates only on the
# household chain: backward-iterate V to its fixed point, forward-
# iterate Λ to stationarity, and the bundle of the two. Anything that
# closes the model around the household block — tatonnement on r / K,
# the MIT-shock transition path, the AR(1) TFP path — is calibration-
# specific and lives in the consuming example or notebook.
#
# Signatures take a single `buffers` bundle (the NamedTuple
# `(; kernels, scratches)` produced by `allocate(hh)`); `V_init` and
# `Λ_init` are kwargs with sensible defaults (zeros for V, uniform for
# Λ), so the common call site reads
#
#     buffers = allocate(hh)
#     (; V, Λ) = solve_steady_state_given_env!(hh, env, buffers)
#
# without any pre-allocation noise. Pass `V_init` / `Λ_init` only when
# warm-starting from a prior solve (e.g., across tatonnement iterations).

# Default initializers #
#----------------------#

# Zero V on the V_start_buffer shape (eltype matches the chain's
# V_start_buffer so AD lifts compose without re-allocation).
_default_V_init(hh::AbstractStage) = zero(V_start_buffer(hh))

# Uniform Λ on the chain's input layout (= first stage's input layout).
function _default_Λ_init(hh::ChainStage)
    layout = first(hh.stages).input_layout
    dims   = layout_size(layout)
    T      = eltype(V_start_buffer(hh))
    return fill(T(inv(prod(dims))), dims)
end


# Inner V fixed point #
#---------------------#

"""
    solve_vfi_steady_state_given_env!(hh, env, buffers;
                                       V_init = zero(V_start_buffer(hh)),
                                       tol = 1e-7, maxiter = 1500)
        -> (; V, iters, converged)

Backward-iterate `V` to its fixed point at `env` by repeatedly applying
`backward!` on the household chain `hh`. `buffers` is the workspace
produced by `allocate(hh)`. `V_init` defaults to zeros; pass a warm
start to skip the cold-V iterations.

Throws an error if `maxiter` is reached before `‖V_new − V‖_∞ ≤ tol`.
"""
function solve_vfi_steady_state_given_env!(hh, env, buffers;
                                            V_init = _default_V_init(hh),
                                            tol::Real    = 1e-7,
                                            maxiter::Int = 4000)
    V = copy(V_init)
    diff = Inf
    iters = 0
    while diff > tol
        V_new = backward!(hh, V, env, buffers)
        diff  = maximum(abs, V_new .- V)
        V .= V_new
        iters += 1
        iters == maxiter && error("solve_vfi_steady_state_given_env!: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (; V, iters, converged = true)
end


# Inner Λ fixed point #
#---------------------#

"""
    solve_lambda_steady_state_given_env!(hh, buffers;
                                          Λ_init = uniform(...),
                                          tol = 1e-6, maxiter = 20_000)
        -> (; Λ, iters, converged)

Forward-iterate `Λ` to its stationary distribution by repeatedly
applying `forward!` on the household chain `hh`. The kernels in
`buffers.kernels` are whatever the most recent `backward!` populated;
the caller is responsible for having seeded them at the env this
iteration is intended for. `Λ_init` defaults to the uniform
distribution over the chain's input layout.

Throws an error if `maxiter` is reached before `‖Λ_new − Λ‖_∞ ≤ tol`.
"""
function solve_lambda_steady_state_given_env!(hh, buffers;
                                               Λ_init = _default_Λ_init(hh),
                                               tol::Real    = 1e-6,
                                               maxiter::Int = 20_000)
    Λ = copy(Λ_init)
    diff = Inf
    iters = 0
    while diff > tol
        Λ_new = forward!(hh, Λ, buffers)
        diff  = maximum(abs, Λ_new .- Λ)
        Λ .= Λ_new
        iters += 1
        iters == maxiter && error("solve_lambda_steady_state_given_env!: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (; Λ, iters, converged = true)
end


# Bundle: V fixed point, then Λ fixed point, at a single env #
#-----------------------------------------------------------#

"""
    solve_steady_state_given_env!(hh, env, buffers;
                                   V_init = zero(V_start_buffer(hh)),
                                   Λ_init = uniform(...),
                                   vfi_tol = 1e-7, vfi_maxiter = 1500,
                                   lambda_tol = 1e-6, lambda_maxiter = 20_000)
        -> (; V, Λ, vfi_iters, lambda_iters)

Bundle of the two single-env inner solves: run `V` to its fixed point
at `env`, then `Λ` to stationarity using the policy that backward
populated in `buffers.kernels`. Returns the converged `(V, Λ)` along
with the inner iteration counts.

`V_init` / `Λ_init` default to a zero-V / uniform-Λ start — most callers
do not need to pass them. Tatonnement loops typically warm-start by
passing the previous outer iterate's `V` / `Λ`.

This helper takes `env` for granted — it does NOT touch any outer loop,
does NOT compute moments, and does NOT update an aggregate state. Wrap
it in a tatonnement loop in the consumer (per-example driver / notebook)
and call `compute_moments(hh, env)` after for moment readout.
"""
function solve_steady_state_given_env!(hh, env, buffers;
                                        V_init = _default_V_init(hh),
                                        Λ_init = _default_Λ_init(hh),
                                        vfi_tol::Real       = 1e-7,
                                        vfi_maxiter::Int    = 1500,
                                        lambda_tol::Real    = 1e-6,
                                        lambda_maxiter::Int = 20_000)
    vfi = solve_vfi_steady_state_given_env!(hh, env, buffers;
                                            V_init  = V_init,
                                            tol     = vfi_tol,
                                            maxiter = vfi_maxiter)
    lam = solve_lambda_steady_state_given_env!(hh, buffers;
                                                Λ_init  = Λ_init,
                                                tol     = lambda_tol,
                                                maxiter = lambda_maxiter)
    return (; V = vfi.V, Λ = lam.Λ,
            vfi_iters = vfi.iters, lambda_iters = lam.iters)
end
