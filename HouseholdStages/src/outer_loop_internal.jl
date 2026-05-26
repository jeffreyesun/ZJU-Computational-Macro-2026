###############################################################
# Outer-loop primitives — Spec/Buffer-keyed                   #
###############################################################
#
# These primitives take a Spec plus a Buffer explicitly and do no
# bookkeeping beyond the iteration itself:
#
#   * No moment computation (caller's job).
#   * No auto-warm-start — `V_init` / `Λ_init` are explicit kwargs
#     with sensible default shapes.
#   * The buffer is mutated in place by the per-iteration
#     `backward!` / `forward!` calls; no separate copy-back.
#
# The Stage-keyed public API lives in `outer_loop.jl`. Same names;
# multiple dispatch separates them.

# Buffer-shape helpers #
#----------------------#

_buffer_V_start(buffer::AbstractStageBuffer) = buffer.V_start
_buffer_V_start(buffer::ChainStageBuffer)    = buffer.stages[1].V_start
_buffer_V_start(buffer::ProductStageBuffer)  = buffer.V_fused

_buffer_Λ_end(buffer::AbstractStageBuffer) = buffer.Λ_end
_buffer_Λ_end(buffer::ChainStageBuffer)    = buffer.stages[end].Λ_end
_buffer_Λ_end(buffer::ProductStageBuffer)  = buffer.Λ_fused

_buffer_input_layout(buffer::AbstractStageBuffer) = buffer.input_layout
_buffer_input_layout(buffer::ChainStageBuffer)    = buffer.input_layout
_buffer_input_layout(buffer::ProductStageBuffer)  = buffer.input_layout

_default_V_init(buffer::AbstractStageBuffer) = zero(_buffer_V_start(buffer))

function _default_Λ_init(buffer::AbstractStageBuffer)
    layout = _buffer_input_layout(buffer)
    dims   = layout_size(layout)
    T      = eltype(_buffer_V_start(buffer))
    return fill(T(inv(prod(dims))), dims)
end


# Inner V fixed point #
#---------------------#

"""
Spec/Buffer-keyed VFI: backward-iterate `V` to its fixed point at
`env` by repeatedly applying `backward!`. Errors on non-convergence.
"""
function solve_vfi_steady_state_given_env!(spec::AbstractStageSpec, env,
                                           buffer::AbstractStageBuffer;
                                           V_init=_default_V_init(buffer),
                                           tol::Real=1e-7,
                                           maxiter::Int=4000)
    V     = copy(V_init)
    diff  = Inf
    iters = 0
    while diff > tol
        V_new = backward!(buffer, spec, V, env)
        diff  = maximum(abs, V_new .- V)
        V .= V_new
        iters += 1
        iters == maxiter && error("solve_vfi_steady_state_given_env!: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (; V, iters, converged=true)
end


# Inner Λ fixed point #
#---------------------#

"""
Spec/Buffer-keyed Λ-iteration: forward-iterate `Λ` to its stationary
distribution. Kernels must have been seated by a prior `backward!`.
"""
function solve_lambda_steady_state_given_env!(spec::AbstractStageSpec,
                                              buffer::AbstractStageBuffer;
                                              Λ_init=_default_Λ_init(buffer),
                                              tol::Real=1e-6,
                                              maxiter::Int=20_000)
    Λ     = copy(Λ_init)
    diff  = Inf
    iters = 0
    while diff > tol
        Λ_new = forward!(buffer, spec, Λ)
        diff  = maximum(abs, Λ_new .- Λ)
        Λ .= Λ_new
        iters += 1
        iters == maxiter && error("solve_lambda_steady_state_given_env!: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (; Λ, iters, converged=true)
end


# Bundle: V then Λ at a single env #
#----------------------------------#

"Spec/Buffer-keyed bundle of the two single-env inner solves."
function solve_steady_state_given_env!(spec::AbstractStageSpec, env,
                                       buffer::AbstractStageBuffer;
                                       V_init=_default_V_init(buffer),
                                       Λ_init=_default_Λ_init(buffer),
                                       vfi_tol::Real=1e-7,
                                       vfi_maxiter::Int=4000,
                                       lambda_tol::Real=1e-6,
                                       lambda_maxiter::Int=20_000)
    vfi = solve_vfi_steady_state_given_env!(spec, env, buffer;
                                            V_init, tol=vfi_tol, maxiter=vfi_maxiter)
    lam = solve_lambda_steady_state_given_env!(spec, buffer;
                                               Λ_init, tol=lambda_tol, maxiter=lambda_maxiter)
    return (; V=vfi.V, Λ=lam.Λ, vfi_iters=vfi.iters, lambda_iters=lam.iters)
end


# Transition path #
#-----------------#

"""
Spec-keyed deterministic transition. Allocates `T` independent
per-period chains (one Buffer per period) so each period's seated
kernel survives the matching `forward!`.
"""
function solve_transition_given_env_path!(spec::ChainStageSpec,
                                          env_path::AbstractVector;
                                          Λ_0::AbstractArray,
                                          V_T::AbstractArray,
                                          layout::StateLayout,
                                          max_inner_iters::Int=1)
    T_steps = length(env_path)
    @assert T_steps >= 1

    # Per-period chains sharing the spec but with fresh buffers each.
    hh_path = [ChainStage(spec, layout) for _ in 1:T_steps]

    dims_V = size(V_T)
    dims_Λ = size(Λ_0)
    Tel    = eltype(V_T)
    V_path = [zeros(Tel, dims_V) for _ in 1:T_steps+1]
    Λ_path = [zeros(Tel, dims_Λ) for _ in 1:T_steps+1]
    copyto!(V_path[T_steps+1], V_T)
    copyto!(Λ_path[1],         Λ_0)

    iters = 0
    for _ in 1:max_inner_iters
        for t in T_steps:-1:1
            V_t = backward!(hh_path[t], V_path[t+1], env_path[t])
            copyto!(V_path[t], V_t)
        end
        for t in 1:T_steps
            Λ_t1 = forward!(hh_path[t], Λ_path[t])
            copyto!(Λ_path[t+1], Λ_t1)
        end
        iters += 1
    end

    moments_path = if isempty(spec.moments)
        [(;) for _ in 1:T_steps]
    else
        [compute_moments(hh_path[t], Λ_path[t+1], env_path[t]) for t in 1:T_steps]
    end

    return (; V_path, Λ_path, moments_path,
            history=(inner_iters=iters,), iters)
end


# Direct-effect Jacobian #
#------------------------#

"""
Spec/Buffer-keyed period-0 direct-effect Jacobian. Reads V_ss / Λ_ss
from the buffer (caller must have populated by a prior
`solve_steady_state_given_env!`), perturbs `env_ss` by ±eps along
each input, FD-computes period-0 direct effects.
"""
function compute_direct_jacobian!(spec::ChainStageSpec, env_ss, T::Int,
                                  buffer::ChainStageBuffer;
                                  inputs=nothing, outputs=nothing,
                                  eps::Real=1e-5)
    @assert !isempty(spec.moments) "compute_direct_jacobian!: chain has no moments attached; call define_moment! first."
    V_ss = copy(_buffer_V_start(buffer))
    Λ_ss = copy(_buffer_Λ_end(buffer))

    in_names  = inputs  === nothing ? Tuple(keys(env_ss))      : Tuple(inputs)
    out_names = outputs === nothing ? Tuple(keys(spec.moments)) : Tuple(outputs)

    jacs = Dict{Tuple{Symbol, Symbol}, Matrix{Float64}}()
    for input in in_names
        m_plus  = _moments_at(spec, buffer, _perturb(env_ss, input, +eps), V_ss, Λ_ss)
        m_minus = _moments_at(spec, buffer, _perturb(env_ss, input, -eps), V_ss, Λ_ss)
        for output in out_names
            curlyY_0 = (m_plus[output] - m_minus[output]) / (2 * eps)
            J = zeros(T, T)
            for t in 1:T
                J[t, t] = curlyY_0
            end
            jacs[(input, output)] = J
        end
    end

    length(jacs) == 1 && return first(values(jacs))
    return jacs
end

"""
Construct a perturbed copy of `env` with field `name` shifted by `amount`.
"""
function _perturb(env, name::Symbol, amount::Real)
    @assert haskey(env, name)
    return merge(env, NamedTuple{(name,)}((getproperty(env, name) + amount,)))
end

"""
Solve the chain at the given env, warm-starting from `V_init` / `Λ_init`,
and return the moment NamedTuple.
"""
function _moments_at(spec::ChainStageSpec, buffer::ChainStageBuffer, env,
                     V_init, Λ_init)
    res = solve_steady_state_given_env!(spec, env, buffer;
                                        V_init=copy(V_init), Λ_init=copy(Λ_init))
    return compute_moments(spec, buffer.output_layout, res.Λ, env)
end
