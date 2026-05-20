######################################################
# Aiyagari MIT Shock — Sequence-Space Utilities Demo  #
######################################################

# Demonstrates `HouseholdStages.expectation_vectors`, `build_F`, and
# `J_from_F` at the Aiyagari steady state on the 3-stage household
# chain `IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`.
#
# `expectation_vectors` iterates the chain's `forward_adjoint!` —
# i.e., K^T applied to the integrand — to produce the t-step
# expectation arrays. Under the new chain, that requires
# `forward_adjoint!` methods on both `WealthChangeStage` and
# `ConsumptionSavingsStage`; see `HouseholdStages/src/lifts/jacobian.jl`.
#
# This script:
#   1. Solves the pre-shock steady state (A = 1) via tatonnement on K.
#   2. Calls `expectation_vectors` for the `K_supplied` integrand
#      `cell -> cell.wealth` over a fixed horizon `T_horizon`.
#   3. Synthesises per-period (𝒴, 𝒟) inputs and assembles the
#      fake-news matrix and sequence-space Jacobian via `build_F` +
#      `J_from_F`.
#   4. Prints the diagonal and a few off-diagonals so the user can
#      eyeball the structure.

include("transition.jl")          # pulls in model.jl and mit_steady_state

using Printf

"""
Run the household-layer sequence-space-Jacobian pipeline at the
pre-shock Aiyagari steady state and print key entries.
"""
function ssj_demo(; T_horizon::Int = 30, p = mit_shock_params,
                    verbose::Bool = true)
    # 1. Pre-shock steady state (A = 1)
    verbose && println("Computing pre-shock steady state (A = 1)…")
    ss     = mit_steady_state(p; A = 1.0, verbosity = 0)
    hh     = ss.hh
    K_ss   = ss.K
    env_ss = (; K = K_ss, r = ss.r, w = ss.w)
    verbose && @printf "  K_ss = %.4f, r = %.4f, w = %.4f\n" K_ss ss.r ss.w

    # Re-prime kernels at the SS so the adjoint methods read a
    # consistent K-operator.
    backward!(hh, ss.V, env_ss)
    forward!(hh, ss.Λ)

    # 2. Expectation vectors for the K_supplied integrand
    verbose && println("\nRunning expectation_vectors over $T_horizon periods…")
    𝓔 = expectation_vectors(hh, cell -> cell.wealth, T_horizon)
    if verbose
        println("  produced $(length(𝓔)) expectation arrays of shape $(size(𝓔[1])).")
        println("  ⟨𝓔[t], Λ_ss⟩ for t = 0..min(5, T_horizon-1):")
        for t in 1:min(6, T_horizon)
            @printf "    t = %d: %.4f\n" (t-1) sum(𝓔[t] .* ss.Λ)
        end
    end

    # 3. Synthesise per-period direct effects.
    dims = size(𝓔[1])
    curlyY = ones(T_horizon)
    curlyD = [zeros(Float64, dims...) for _ in 1:T_horizon]
    mid_w_idx = div(dims[1], 2)
    for t in 1:T_horizon
        curlyD[t][mid_w_idx, 1]   = 1.0
        curlyD[t][mid_w_idx, end] = -1.0
    end

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

    return (; K_ss, V_ss = ss.V, Λ_ss = ss.Λ, env_ss,
             𝓔, F, J)
end

# Run when executed as a script #
#-------------------------------#

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running HouseholdStages sequence-space-utilities demo on the new 3-stage chain…")
    @time res = ssj_demo(T_horizon = 30)
    println("\nDone.")
end
