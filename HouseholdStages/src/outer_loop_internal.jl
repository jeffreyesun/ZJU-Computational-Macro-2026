###############################################################
# Outer-loop primitives — Spec/Buffer-keyed                   #
###############################################################
#
# This file holds the *internal* primitives. They take a Spec plus
# a Buffer explicitly, follow the same dispatch convention as
# `backward!(spec, V_end, env, buffer)` / `forward!(spec, Λ, buffer)`,
# and do no bookkeeping beyond the iteration itself:
#
#   * No moment computation (the caller's job).
#   * No auto-warm-start from buffer state — `V_init` / `Λ_init` are
#     explicit kwargs with sensible default shapes.
#   * The buffer is mutated in place (kernels seated, V_start / Λ_end
#     filled by the final iteration's `backward!` / `forward!`), but
#     no separate copy-back step is performed.
#
# The Stage-keyed public API lives in `outer_loop.jl`. Same function
# names; Julia's multiple dispatch separates the two methods.

# Buffer-shape helpers #
#----------------------#

# Buffer-side analog of `V_start_buffer(stage)`, used to derive default
# shapes for `V_init` / `Λ_init` without going through the bundled Stage.

_buffer_V_start(buffer::AbstractStageBuffer)        = buffer.V_start
_buffer_V_start(buffer::ChainStageBuffer)           = buffer.stages[1].V_start
_buffer_V_start(buffer::ProductStageBuffer)         = buffer.V_fused

_buffer_Λ_end(buffer::AbstractStageBuffer)          = buffer.Λ_end
_buffer_Λ_end(buffer::ChainStageBuffer)             = buffer.stages[end].Λ_end
_buffer_Λ_end(buffer::ProductStageBuffer)           = buffer.Λ_fused


# Default V_init: zeros matching the buffer's V_start shape.
_default_V_init(buffer::AbstractStageBuffer) = zero(_buffer_V_start(buffer))

# Default Λ_init: uniform over the spec's input layout.
function _default_Λ_init(spec::AbstractStageSpec, buffer::AbstractStageBuffer)
    layout = spec.input_layout
    dims   = layout_size(layout)
    T      = eltype(_buffer_V_start(buffer))
    return fill(T(inv(prod(dims))), dims)
end

function _default_Λ_init(spec::ChainStageSpec, buffer::ChainStageBuffer)
    layout = first(spec.stages).input_layout
    dims   = layout_size(layout)
    T      = eltype(_buffer_V_start(buffer))
    return fill(T(inv(prod(dims))), dims)
end


# Inner V fixed point #
#---------------------#

"""
    solve_vfi_steady_state_given_env!(spec, env, buffer;
                                       V_init = zero(_buffer_V_start(buffer)),
                                       tol = 1e-7, maxiter = 4000)
        -> (; V, iters, converged)

Spec/Buffer-keyed primitive: backward-iterate `V` to its fixed point at
`env` by repeatedly applying `backward!(spec, V, env, buffer)`. The
buffer's kernel is seated by the final `backward!`; the caller owns
any further bookkeeping (e.g., copying V into the buffer's V_start
slot for warm-starting).

Errors if `maxiter` is reached before `‖V_new − V‖_∞ ≤ tol`.
"""
function solve_vfi_steady_state_given_env!(spec::AbstractStageSpec, env,
                                           buffer::AbstractStageBuffer;
                                           V_init       = _default_V_init(buffer),
                                           tol::Real    = 1e-7,
                                           maxiter::Int = 4000)
    V    = copy(V_init)
    diff = Inf
    iters = 0
    while diff > tol
        V_new = backward!(spec, V, env, buffer)
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
    solve_lambda_steady_state_given_env!(spec, buffer;
                                          Λ_init = uniform,
                                          tol = 1e-6, maxiter = 20_000)
        -> (; Λ, iters, converged)

Spec/Buffer-keyed primitive: forward-iterate `Λ` to its stationary
distribution by repeatedly applying `forward!(spec, Λ, buffer)`. The
buffer's kernels must have been seated by a prior `backward!` call
at the env this iteration is intended for — this primitive does not
touch them.

Errors if `maxiter` is reached before `‖Λ_new − Λ‖_∞ ≤ tol`.
"""
function solve_lambda_steady_state_given_env!(spec::AbstractStageSpec,
                                              buffer::AbstractStageBuffer;
                                              Λ_init       = _default_Λ_init(spec, buffer),
                                              tol::Real    = 1e-6,
                                              maxiter::Int = 20_000)
    Λ    = copy(Λ_init)
    diff = Inf
    iters = 0
    while diff > tol
        Λ_new = forward!(spec, Λ, buffer)
        diff  = maximum(abs, Λ_new .- Λ)
        Λ .= Λ_new
        iters += 1
        iters == maxiter && error("solve_lambda_steady_state_given_env!: failed to converge in $maxiter iterations (last diff = $diff)")
    end
    return (; Λ, iters, converged = true)
end


# Bundle: V then Λ at a single env #
#----------------------------------#

"""
    solve_steady_state_given_env!(spec, env, buffer;
                                   V_init, Λ_init,
                                   vfi_tol, vfi_maxiter,
                                   lambda_tol, lambda_maxiter)
        -> (; V, Λ, vfi_iters, lambda_iters)

Spec/Buffer-keyed primitive bundling the two single-env inner solves.
**Does not** compute moments and **does not** write V/Λ back into the
buffer beyond what the iterations themselves leave there. The
Stage-keyed wrapper in `outer_loop.jl` adds those steps.
"""
function solve_steady_state_given_env!(spec::AbstractStageSpec, env,
                                       buffer::AbstractStageBuffer;
                                       V_init              = _default_V_init(buffer),
                                       Λ_init              = _default_Λ_init(spec, buffer),
                                       vfi_tol::Real       = 1e-7,
                                       vfi_maxiter::Int    = 4000,
                                       lambda_tol::Real    = 1e-6,
                                       lambda_maxiter::Int = 20_000)
    vfi = solve_vfi_steady_state_given_env!(spec, env, buffer;
                                            V_init  = V_init,
                                            tol     = vfi_tol,
                                            maxiter = vfi_maxiter)
    lam = solve_lambda_steady_state_given_env!(spec, buffer;
                                               Λ_init  = Λ_init,
                                               tol     = lambda_tol,
                                               maxiter = lambda_maxiter)
    return (; V = vfi.V, Λ = lam.Λ,
            vfi_iters = vfi.iters, lambda_iters = lam.iters)
end


# Transition path #
#-----------------#

"""
    solve_transition_given_env_path!(spec::ChainStageSpec, env_path;
                                      Λ_0, V_T, max_inner_iters = 1)
        -> (; V_path, Λ_path, moments_path, history, iters)

Spec-keyed primitive for a deterministic transition. Allocates `T`
independent chain copies sharing `spec` (one Buffer per period), runs
a backward sweep `t = T:-1:1` then a forward sweep `t = 1:T`, returning
the V/Λ paths plus per-period moments.

Per-period chains share the chain's Spec but each has its own Buffer,
so each period's `backward!`-produced kernel is preserved for the
matching `forward!`. The structural fix for the L05 footgun.

Note: this primitive does NOT take a single Buffer argument — the
function's job is to allocate per-period buffers from the Spec, so
"the buffer" is intrinsically plural. The Stage-keyed wrapper in
`outer_loop.jl` is a thin delegate.
"""
function solve_transition_given_env_path!(spec::ChainStageSpec,
                                          env_path::AbstractVector;
                                          Λ_0::AbstractArray,
                                          V_T::AbstractArray,
                                          max_inner_iters::Int = 1)
    T_steps = length(env_path)
    T_steps ≥ 1 ||
        error("solve_transition_given_env_path!: env_path must be non-empty")

    # Per-period chains sharing the spec but with fresh buffers each.
    hh_path = [ChainStage(spec, allocate(spec)) for _ in 1:T_steps]

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
            history = (inner_iters = iters,),
            iters)
end


# Direct-effect Jacobian #
#------------------------#

"""
    compute_direct_jacobian!(spec::ChainStageSpec, env_ss, T, buffer;
                              inputs = nothing, outputs = nothing,
                              eps = 1e-5)
        -> Jacobian (Matrix or Dict{(input, output), Matrix})

Spec/Buffer-keyed primitive: reads V_ss / Λ_ss from the buffer (which
the caller must have populated by a prior `solve_steady_state_given_env!`),
perturbs `env_ss` by `±eps` along each named input, re-solves at each
perturbed env, computes finite-difference period-0 direct effects, and
writes them on the diagonal of a `T × T` matrix. Off-diagonal entries
are zero by construction — this is the period-0 direct effect, not the
full sequence-space Jacobian.

For one (input, output) pair, returns a `T × T` matrix; otherwise a
`Dict{Tuple{Symbol, Symbol}, Matrix}` keyed by (input, output).
"""
function compute_direct_jacobian!(spec::ChainStageSpec, env_ss, T::Int,
                                  buffer::ChainStageBuffer;
                                  inputs    = nothing,
                                  outputs   = nothing,
                                  eps::Real = 1e-5)
    isempty(spec.moments) &&
        error("compute_direct_jacobian!: chain has no moments attached; call define_moment! first.")
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

    if length(jacs) == 1
        return first(values(jacs))
    end
    return jacs
end

"""CLAUDE
Construct a perturbed copy of `env` with field `name` shifted by
`amount`. Other fields pass through unchanged.
"""
function _perturb(env, name::Symbol, amount::Real)
    @assert haskey(env, name) "perturb: env has no field :$name"
    pairs = NamedTuple{(name,)}((getproperty(env, name) + amount,))
    return merge(env, pairs)
end

"""CLAUDE
Solve the chain at the given env (warm-starting from V_init / Λ_init via
the Spec/Buffer-keyed primitive) and return the moment NamedTuple.
Used by the FD direct-effect step in `compute_direct_jacobian!`.
"""
function _moments_at(spec::ChainStageSpec, buffer::ChainStageBuffer, env,
                     V_init, Λ_init)
    res = solve_steady_state_given_env!(spec, env, buffer;
                                        V_init = copy(V_init),
                                        Λ_init = copy(Λ_init))
    return compute_moments(spec, res.Λ, env)
end
