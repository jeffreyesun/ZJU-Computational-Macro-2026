######################################################
# Aiyagari MIT Shock — Sequence-Space Utilities Demo  #
######################################################

# Demonstrates `HouseholdStages.expectation_vectors`, `build_F`, and
# `J_from_F` at the Aiyagari steady state on the 3-stage household
# chain `IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings`.
#
# `expectation_vectors` iterates the chain's `forward_adjoint!` —
# i.e., K^T applied to the integrand — to produce the t-step
# expectation arrays. Under the new chain, that requires
# `forward_adjoint!` methods on both `WealthChange` and
# `ConsumptionSavings`; they were added in the same 2026-05-15 sub-
# agent dispatch as this example's refactor (see
# `notes/overnight_2026-05-15_mit_shock_refactor.md` and
# `HouseholdStages/src/lifts/jacobian.jl`).
#
# This script:
#   1. Solves the steady state via tatonnement on K (`mit_steady_state`).
#   2. Calls `expectation_vectors` for the `K_supplied` integrand
#      `cell -> cell.wealth` over a fixed horizon `T_horizon`.
#   3. Synthesises per-period (𝒴, 𝒟) inputs and assembles the
#      fake-news matrix and sequence-space Jacobian via `build_F` +
#      `J_from_F`.
#   4. Prints the diagonal and a few off-diagonals so the user can
#      eyeball the structure.
#
# Hard-argmax `ConsumptionSavings` produces a step-function policy, so
# a small-ε finite-difference cross-check is zero by construction (the
# integer policy is locally invariant). The per-stage reverse-mode
# adjoints handle this via the envelope theorem; a meaningful FD
# validation would require a smoothed savings policy
# (`LogitChoice`-with-moderate-ε) — that's a natural next example
# variant, not covered here.

include("transition.jl")          # pulls in model.jl and mit_steady_state

using Printf

"""
Run the household-layer sequence-space-Jacobian pipeline at the
Aiyagari steady state and print key entries.

`T_horizon` is the number of periods of K-transpose iteration. The
fake-news matrix `F` has shape `(T_horizon, T_horizon)` and the
returned Jacobian `J = J_from_F(F)` is the same.
"""
function ssj_demo(; T_horizon::Int = 30, p = mit_shock_params,
                    verbose::Bool = true)
    # 1. Steady state
    verbose && println("Computing steady state…")
    ss = mit_steady_state(p; verbosity = 0)
    hh        = ss.hh
    caches    = ss.caches
    scratches = ss.scratches
    K_ss      = ss.K
    env_ss    = (;K = K_ss, r = ss.r, w = ss.w)
    verbose && @printf "  K_ss = %.4f, r = %.4f, w = %.4f\n" K_ss ss.r ss.w

    # Re-prime caches at the SS so the adjoint methods read a
    # consistent K-operator (the steady-state tatonnement's last
    # backward/forward calls were at the converged K, but to be
    # explicit we redo them here).
    backward!(hh, ss.V, env_ss, caches, scratches)
    forward!(hh, ss.Λ, caches, scratches)

    # 2. Expectation vectors for the K_supplied integrand
    verbose && println("\nRunning expectation_vectors over $T_horizon periods…")
    𝓔 = expectation_vectors(hh, cell -> cell.wealth,
                            T_horizon, caches, scratches)
    if verbose
        println("  produced $(length(𝓔)) expectation arrays of shape $(size(𝓔[1])).")
        # Inner-product check: ⟨𝓔[t], Λ_ss⟩ ≈ K_ss for all t, because
        # under K-transpose iteration starting from cell.wealth, the
        # t-step expectation viewed against the stationary Λ_ss is the
        # K_ss moment up to numerical noise.
        println("  ⟨𝓔[t], Λ_ss⟩ for t = 0..min(5, T_horizon-1):")
        for t in 1:min(6, T_horizon)
            @printf "    t = %d: %.4f\n" (t-1) sum(𝓔[t] .* ss.Λ)
        end
    end

    # 3. Synthesise per-period direct effects.
    # 𝒴_t = curlyY[t] — direct contemporaneous aggregate-output impact
    # of a shock at time t (modelling slot for SSJ Step 1's "non-
    # distributional" piece). Set to 1 for a unit shock signal.
    # 𝒟_t = curlyD[t] — direct perturbation to the end-of-period
    # distribution induced by the shock. For a demo we synthesise a
    # zero-sum bump that shifts a unit of mass from the lowest-income
    # tail to the highest-income tail at the median wealth cell —
    # consistent with a shock that re-orients the income process.
    dims = size(𝓔[1])
    curlyY = ones(T_horizon)
    curlyD = [zeros(Float64, dims...) for _ in 1:T_horizon]
    mid_w_idx = div(dims[1], 2)
    for t in 1:T_horizon
        curlyD[t][mid_w_idx, 1]   = 1.0
        curlyD[t][mid_w_idx, end] = -1.0
    end

    # `build_F` consumes `curlyE = 𝓔[2:end]` (length T_horizon - 1) and
    # produces F with `length(curlyE) + 1 = T_horizon` rows. Matches the
    # convention in `HouseholdStages.sequence_space.jl`.
    𝓔_vec = 𝓔[2:end]
    F = build_F(curlyY, curlyD, 𝓔_vec)
    J = J_from_F(F)

    if verbose
        println("\nbuild_F + J_from_F:")
        @printf "  F : %s ; J : %s\n" size(F) size(J)
        println("  Diagonal of J for t = 0..min(7, T_horizon-1):")
        for i in 1:min(8, size(J, 1))
            @printf "    J[%d, %d] = %+.6f\n" (i-1) (i-1) J[i, i]
        end
        println("  First column of J (response to t = 0 shock):")
        for i in 1:min(6, size(J, 1))
            @printf "    J[%d, 0] = %+.6f\n" (i-1) J[i, 1]
        end
    end

    return (;K_ss, V_ss = ss.V, Λ_ss = ss.Λ, env_ss,
             𝓔, F, J)
end

# Run when executed as a script #
#-------------------------------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running HouseholdStages sequence-space-utilities demo on the new 3-stage chain…")
    @time res = ssj_demo(T_horizon = 30)
    println("\nDone.")
end
