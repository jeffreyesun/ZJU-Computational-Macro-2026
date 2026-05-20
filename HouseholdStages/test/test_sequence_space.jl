using Test
using HouseholdStages

@testset "J_from_F — recursive cumulation by hand" begin
    # F[s, t]: row 1 = direct effect at shock time t; row > 1 = effect s
    # periods after the shock. J cumulates anti-diagonals.
    F = Float64[
        1  2  3 ;       # direct effects at t = 1, 2, 3
        0  0  0 ;       # 1-period-ahead distribution-mediated effects
        0  0  0 ;       # 2-period-ahead
    ]
    # With no distribution-mediated effects, the cumulation only adds
    # the row-1 direct effects to subsequent rows along anti-diagonals.
    # Specifically: J[s+1, t] += J[s, t-1].
    J = J_from_F(F)
    # Verify the anti-diagonal cumulation explicitly:
    @test J[1, :] == [1, 2, 3]    # direct effects unchanged
    @test J[2, 1] == 0
    @test J[2, 2] == 1            # added from J[1, 1] = 1
    @test J[2, 3] == 2            # added from J[1, 2] = 2
    @test J[3, 1] == 0
    @test J[3, 2] == 0
    @test J[3, 3] == 1            # added from J[2, 2] = 1
end

@testset "J_from_F — non-trivial distribution-mediated effects" begin
    F = Float64[
        1  0  0 ;
        2  0  0 ;
        3  0  0 ;
    ]
    J = J_from_F(F)
    @test J[1, :] == [1, 0, 0]
    @test J[2, :] == [2, 1, 0]    # row 2: F[2] + shifted-J[1]
    @test J[3, :] == [3, 2, 1]    # row 3: F[3] + shifted-J[2]
end

@testset "build_F — direct effects in row 1; outer products in rows ≥ 2" begin
    curlyY = Float64[1.0, 2.0]
    curlyD = [Float64[0.5, 0.5], Float64[1.0, 0.0]]  # one per shock time
    curlyE = [Float64[2.0, 0.0]]                       # one period of E
    # F has shape (T_lookahead, T) = (2, 2).
    F = build_F(curlyY, curlyD, curlyE)
    @test F[1, :] == [1.0, 2.0]
    # F[2, 1] = sum(curlyE[1] .* curlyD[1]) = 2.0 * 0.5 + 0 * 0.5 = 1.0
    # F[2, 2] = sum(curlyE[1] .* curlyD[2]) = 2.0 * 1.0 + 0 * 0.0 = 2.0
    @test F[2, 1] ≈ 1.0
    @test F[2, 2] ≈ 2.0
end

@testset "expectation_vectors — pure Markov chain at SS" begin
    # Two-state Markov with transition matrix P (rows = today, cols =
    # tomorrow). For pure Markov, K = P^T (forward); K^T = P (backward).
    # E_t[integrand](s) = P^t · integrand applied as a vector.
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.0, 1.0])))
    chain = MarkovStage(layout; axis = :z, transition = P)
    # Seed kernels at the steady state via a backward call.
    backward!(chain, zeros(2), NamedTuple())
    # Integrand: identity on z (so E_t[z | s_0 = s] is the t-step
    # forward expected value of z).
    integrand = cell -> cell.z
    Es = expectation_vectors(chain, integrand, 3)
    @test length(Es) == 3
    # E_0 = [0.0, 1.0] (identity).
    @test Es[1] ≈ [0.0, 1.0]
    # E_1[s] = sum_{s'} P[s, s'] z[s'] = P[s, 1]·0 + P[s, 2]·1 = P[s, 2].
    @test Es[2] ≈ [P[1, 2], P[2, 2]]
    # E_2 = P · E_1 (matrix product).
    @test Es[3] ≈ P * Es[2] atol = 1e-12
end

@testset "expectation_vectors — chain (Markov ∘ Identity)" begin
    P = [0.8 0.2; 0.2 0.8]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.0, 1.0])))
    s1 = MarkovStage(layout; axis = :z, transition = P)
    s2 = IdentityStage(layout)
    chain = s1 ∘ s2
    # Seed kernels.
    backward!(chain, zeros(2), NamedTuple())
    integrand = cell -> cell.z
    Es = expectation_vectors(chain, integrand, 2)
    @test length(Es) == 2
    @test Es[1] ≈ [0.0, 1.0]
    # The chain's K^T = (IdentityStage K^T) · (MarkovStage K^T) = I · P.
    @test Es[2] ≈ P * Es[1] atol = 1e-12
end
