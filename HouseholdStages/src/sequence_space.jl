###################################
# Aggregate-Jacobian Utilities    #
# (fake-news Steps 2, 3, 4)       #
###################################
#
# Outer-loop accounting steps in SSJ's fake-news algorithm
# (stages_paper/notes/proofs.md Cell 1.7). Convert per-stage tangents
# into sequence-space Jacobians at the steady state. Step 1 (per-stage
# tangent propagation via F_J_fwd) lives in `src/lifts/jacobian.jl`
# via `lift_jacobian(stage; mode=:forward)`.

"""
    expectation_vectors(chain, integrand, T) -> Vector{Array}

Compute the time-t expectation of `integrand` viewed from the
beginning of the period, for t = 0, 1, ..., T-1, at the steady state
whose kernels were computed by a prior `backward!` call.

`integrand` is a function `cell -> value`; it is evaluated at every
cell of the chain's terminal output layout to produce the t=0 array.
For t > 0, the result is obtained by applying the chain's K-transpose
(i.e., the chain's `forward_adjoint!`) to the previous time's array.

The chain's kernels (stored in `chain.buffer`) should have been
populated by a prior `backward!(chain, V_terminal, env_ss)` call at
the steady state. The K used in the K-transpose iteration is the
kernel materialized at that backward-pass evaluation point. (No env
argument here — env was consumed by the prior backward pass.)

This corresponds to Step 2 of SSJ's fake-news algorithm.

Returns a Vector of length T, where each element is the t-th
expectation array on the chain's terminal-output layout.
"""
function expectation_vectors(chain::AbstractStage, integrand::Function, T::Int)
    @assert T ≥ 1 "expectation_vectors: T must be at least 1"
    out_layout = _terminal_out_layout(chain)
    dims = layout_size(out_layout)
    Tnum = eltype_from_chain(chain)
    E0 = zeros(Tnum, dims...)
    for (idx, cell) in cells(out_layout)
        ci = CartesianIndex(values(idx))
        E0[ci] = integrand(cell)
    end
    results = Vector{Array{Tnum, length(dims)}}()
    push!(results, copy(E0))
    E_prev = E0
    for _ in 2:T
        # K-transpose acts on functions; `forward_adjoint!` realizes
        # this (the VJP of forward = K maps a measure-sensitivity by
        # K-transpose). The bundled-stage adjoint reads kernels from
        # `chain.buffer` and is keyed on Spec internally.
        E_next = forward_adjoint!(chain, E_prev)
        push!(results, copy(E_next))
        E_prev = E_next
    end
    return results
end

# Terminal output layout — used to size the integrand-broadcast array.
_terminal_out_layout(s::AbstractStage) = s.spec.output_layout
_terminal_out_layout(c::ChainStage)    = c.spec.out_layout

# Eltype helper: fish out the buffer eltype of the chain's first stage.
eltype_from_chain(chain::ChainStage) = eltype(chain.buffer.stages[1].V_start)
eltype_from_chain(s::AbstractStage)  = eltype(s.buffer.V_start)

"""
    J_from_F(F::AbstractMatrix) -> Matrix

Recursively cumulate the fake-news matrix F into a sequence-space
Jacobian J. Performs the operation `J[s+1, t] += J[s, t-1]` along
anti-diagonals.

F is expected to have shape (T_lookahead, T), where rows index
time-since-shock (row 1 being the contemporaneous direct effect)
and columns index the shock time. The returned J has the same shape.

This corresponds to Step 4 of SSJ's fake-news algorithm.
"""
function J_from_F(F::AbstractMatrix)
    T_lookahead, T = size(F)
    J = copy(F)
    for t in 2:T
        for s in 1:(T_lookahead - 1)
            J[s + 1, t] += J[s, t - 1]
        end
    end
    return J
end

"""
    build_F(curlyY::AbstractVector, curlyD::AbstractVector,
            curlyE::AbstractVector) -> Matrix

Assemble the fake-news matrix from:

  - `curlyY[t]` — direct aggregate-output impact of a shock at time `t`
    (Step 1 of the fake-news algorithm).
  - `curlyD[t]` — perturbation to the end-of-period distribution induced
    by a shock at time `t` (also Step 1).
  - `curlyE[s]` — expectation vectors `s` periods ahead (Step 2 above).

Returns F of shape (length(curlyE) + 1, length(curlyY)). Row 1 is
the direct effect `curlyY`; rows >= 2 are the distribution-mediated
effects `F[s+1, t] = sum(curlyE[s] .* curlyD[t])`.

Step 3 of SSJ's fake-news algorithm.
"""
function build_F(curlyY::AbstractVector, curlyD::AbstractVector,
                 curlyE::AbstractVector)
    T = length(curlyY)
    T_lookahead = length(curlyE) + 1
    @assert length(curlyD) == T "curlyD must have the same length as curlyY (one per shock time)"
    F = zeros(eltype(curlyY), T_lookahead, T)
    for t in 1:T
        F[1, t] = curlyY[t]
        for s in eachindex(curlyE)
            F[s + 1, t] = sum(curlyE[s] .* curlyD[t])
        end
    end
    return F
end
