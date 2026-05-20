###############################################################
# Outer-loop public API — Stage-keyed                          #
###############################################################
#
# Public layer of the household-block computation surface. Every
# function here is the bundled-Stage counterpart to a Spec/Buffer-keyed
# primitive in `outer_loop_internal.jl`. Both methods share the same
# function name; Julia's multiple dispatch routes the call.
#
# At the per-stage level (`backward!`, `forward!`) the Stage-keyed
# method is a pure one-line delegate over the Spec-keyed primary. The
# outer-loop helpers here mostly follow that convention (V-only,
# Λ-only, transition, Jacobian are thin delegates), with one
# substantive exception: the `solve_steady_state_given_env!` bundle
# at `ChainStage` level *absorbs the bookkeeping* — warm-start from
# the chain's buffer state, write converged V/Λ back, compute moments,
# return caller-safe copies. Per-stage `backward!`/`forward!` have no
# bookkeeping to absorb, so they need no analogous logic; at the
# outer-loop layer bookkeeping has to land somewhere, and the public
# entry is the natural home.
#
# Closing-the-model logic (tatonnement on K / r, AR(1) shock
# generation) stays with the consumer.


# Inner V fixed point — public delegate #
#---------------------------------------#

"""
    solve_vfi_steady_state_given_env!(stage, env;
                                       V_init = nothing,
                                       tol = 1e-7, maxiter = 4000)
        -> (; V, iters, converged)

Stage-keyed public form of the V-only fixed-point iteration. Warm-
starts V from `V_start_buffer(stage)` if non-zero (or from the
provided `V_init` kwarg if given), runs the Spec/Buffer-keyed
primitive, writes the converged V back to the buffer for the next
call's warm-start, and returns a caller-safe copy.
"""
function solve_vfi_steady_state_given_env!(stage::AbstractStage, env;
                                           V_init       = nothing,
                                           tol::Real    = 1e-7,
                                           maxiter::Int = 4000)
    V0 = if V_init === nothing
        V_buf = copy(V_start_buffer(stage))
        all(iszero, V_buf) ? zero(V_start_buffer(stage)) : V_buf
    else
        V_init
    end
    res = solve_vfi_steady_state_given_env!(stage.spec, env, stage.buffer;
                                            V_init = V0, tol, maxiter)
    copyto!(V_start_buffer(stage), res.V)
    return (; V = copy(res.V), iters = res.iters, converged = res.converged)
end


# Inner Λ fixed point — public delegate #
#---------------------------------------#

"""
    solve_lambda_steady_state_given_env!(stage;
                                          Λ_init = nothing,
                                          tol = 1e-6, maxiter = 20_000)
        -> (; Λ, iters, converged)

Stage-keyed public form of the Λ-only fixed-point iteration. `Λ_init`
defaults to the uniform distribution over the chain's input layout —
Λ converges fast from uniform and the buffer's Λ_end slot does not
make a meaningful warm-start point (it can hold half-iterated state).
Writes the converged Λ back to the buffer; returns a caller-safe copy.
"""
function solve_lambda_steady_state_given_env!(stage::AbstractStage;
                                              Λ_init       = nothing,
                                              tol::Real    = 1e-6,
                                              maxiter::Int = 20_000)
    Λ0 = if Λ_init === nothing
        _default_Λ_init(stage.spec, stage.buffer)
    else
        Λ_init
    end
    res = solve_lambda_steady_state_given_env!(stage.spec, stage.buffer;
                                               Λ_init = Λ0, tol, maxiter)
    copyto!(Λ_end_buffer(stage), res.Λ)
    return (; Λ = copy(res.Λ), iters = res.iters, converged = res.converged)
end


# Bundle: V then Λ at a single env, with bookkeeping #
#----------------------------------------------------#

"""
    solve_steady_state_given_env!(stage::ChainStage, env;
                                   V_init = nothing, Λ_init = nothing,
                                   vfi_tol, vfi_maxiter,
                                   lambda_tol, lambda_maxiter)
        -> (; V, Λ, moments, history, iters)

User-facing entry point for "solve the household block at this env."
Wraps the Spec/Buffer-keyed primitive in
`outer_loop_internal.jl` with bookkeeping the primitive deliberately
omits:

  * Warm-start `V` from `V_start_buffer(stage)` if non-zero (so a
    second call at a perturbed env reuses the previous solution).
  * Write converged V / Λ into the chain's buffer for the next call.
  * Compute moments (`compute_moments(stage, Λ, env)`) if the chain has
    any attached; empty NamedTuple otherwise.
  * Return caller-safe copies of V and Λ (the originals live in the
    buffer; the caller can hold these across subsequent calls).

History reports the V- and Λ-iteration counts; `iters` is their sum.
"""
function solve_steady_state_given_env!(stage::ChainStage, env;
                                       vfi_tol::Real       = 1e-7,
                                       vfi_maxiter::Int    = 4000,
                                       lambda_tol::Real    = 1e-6,
                                       lambda_maxiter::Int = 20_000,
                                       V_init              = nothing,
                                       Λ_init              = nothing)
    spec, buffer = stage.spec, stage.buffer

    V0 = if V_init === nothing
        V_buf = copy(V_start_buffer(stage))
        all(iszero, V_buf) ? _default_V_init(buffer) : V_buf
    else
        V_init
    end
    Λ0 = Λ_init === nothing ? _default_Λ_init(spec, buffer) : Λ_init

    res = solve_steady_state_given_env!(spec, env, buffer;
                                        V_init = V0, Λ_init = Λ0,
                                        vfi_tol, vfi_maxiter,
                                        lambda_tol, lambda_maxiter)
    V, Λ = res.V, res.Λ

    copyto!(V_start_buffer(stage), V)
    copyto!(Λ_end_buffer(stage),   Λ)

    moments = isempty(spec.moments) ? (;) : compute_moments(spec, Λ, env)
    return (; V = copy(V), Λ = copy(Λ),
            moments,
            history = (vfi_iters = res.vfi_iters,
                       lambda_iters = res.lambda_iters),
            iters = res.vfi_iters + res.lambda_iters)
end


# Transition path — public delegate #
#-----------------------------------#

"""
    solve_transition_given_env_path!(stage::ChainStage, env_path;
                                      Λ_0, V_T, max_inner_iters = 1)
        -> (; V_path, Λ_path, moments_path, history, iters)

Stage-keyed public form of the transition driver. Thin delegate to
the Spec-keyed primitive in `outer_loop_internal.jl`, which allocates
per-period chains internally and does the backward + forward sweep.
Per-period buffers are an implementation detail of the primitive —
there is no single "buffer" to thread, so the Stage-keyed surface
just forwards `stage.spec`.

Boundary conditions:
  * `Λ_0` — initial distribution at the start of period 1.
  * `V_T` — terminal continuation value at the end of period T.
"""
solve_transition_given_env_path!(stage::ChainStage, env_path::AbstractVector;
                                 Λ_0::AbstractArray,
                                 V_T::AbstractArray,
                                 max_inner_iters::Int = 1) =
    solve_transition_given_env_path!(stage.spec, env_path;
                                     Λ_0, V_T, max_inner_iters)


# Direct-effect Jacobian — public delegate #
#------------------------------------------#

"""
    compute_direct_jacobian!(stage::ChainStage, env_ss, T;
                              inputs = nothing, outputs = nothing,
                              eps = 1e-5)
        -> Jacobian (Matrix or Dict{(input, output), Matrix})

Stage-keyed public form. Thin delegate to the Spec/Buffer-keyed
primitive in `outer_loop_internal.jl`, which reads `V_ss`/`Λ_ss`
from the buffer (the caller should typically have just called
`solve_steady_state_given_env!` on the same stage at `env_ss`).

**Direct-effect only.** The function name advertises the scope: this
wrapper computes only the period-0 direct effect of an env
perturbation (`curlyY_0` via two-sided finite differences at the
steady state) and writes it on the diagonal `s = t`. Off-diagonal
entries are zero. It does NOT compute the full fake-news matrix or
the distribution-mediated effects (`curlyD`, `curlyE`).

For the real sequence-space Jacobian, call `expectation_vectors(hh,
integrand, T)`, `build_F(curlyY, curlyD, curlyE)`, and `J_from_F(F)`
directly. See `examples/aiyagari_mit_shock/ssj.jl` for a worked
example.

Treat the returned matrix as a first-order direct-effect diagnostic
only; mistaking it for a real fake-news Jacobian would produce wrong
IRFs.
"""
compute_direct_jacobian!(stage::ChainStage, env_ss, T::Int;
                         inputs    = nothing,
                         outputs   = nothing,
                         eps::Real = 1e-5) =
    compute_direct_jacobian!(stage.spec, env_ss, T, stage.buffer;
                             inputs, outputs, eps)
