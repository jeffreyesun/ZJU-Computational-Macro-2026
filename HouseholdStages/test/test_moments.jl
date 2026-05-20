using Test
using HouseholdStages

@testset "compute_moments — mean wealth on a uniform distribution" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0, 3.0, 4.0])),
        StateAxis(:income, [0.5, 1.5]),
    )
    chain = MarkovStage(layout; axis = :income, transition = P)
    mc = define_moments!(ChainStage((chain,));
        avg_wealth = at_end(integrand = :wealth, reduce = sum),
    )

    # Use a uniform distribution; one forward step keeps it uniform under P.
    Λ_start = fill(1 / 8, 4, 2)
    Λ_end = forward!(mc, Λ_start)
    moments = compute_moments(mc, Λ_end, NamedTuple())
    # Σ_w wealth * Σ_z (1/8) = (1+2+3+4) / 4 = 2.5
    @test moments.avg_wealth ≈ 2.5 atol = 1e-12
end

@testset "compute_moments — multiple specs in one call" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0, 3.0])),
        StateAxis(:income, [0.5, 1.5]),
    )
    chain = MarkovStage(layout; axis = :income, transition = P)
    mc = define_moments!(ChainStage((chain,));
        K = at_end(integrand = :wealth, reduce = sum),
        N = at_end(integrand = :income, reduce = sum),
    )
    Λ_start = fill(1 / 6, 3, 2)
    Λ_end = forward!(mc, Λ_start)
    moments = compute_moments(mc, Λ_end, NamedTuple())
    @test moments.K ≈ 2.0 atol = 1e-12  # (1+2+3)/3
    @test moments.N ≈ 1.0 atol = 1e-12  # (0.5 + 1.5)/2
end

@testset "compute_moments — env-dependent integrand evaluated at call time" begin
    layout = StateLayout(StateAxis(:wealth, continuous_grid([1.0, 2.0])))
    s = IdentityStage(layout)
    mc = define_moments!(ChainStage((s,));
        scaled_wealth = at_end(
            integrand = (cell; env) -> cell.wealth * env.scale,
            reduce    = sum,
        ),
    )
    Λ_start = [0.5, 0.5]
    Λ_end = forward!(mc, Λ_start)
    moments = compute_moments(mc, Λ_end, (scale = 3.0,))
    # 1.0 * 0.5 * 3 + 2.0 * 0.5 * 3 = 4.5
    @test moments.scaled_wealth ≈ 4.5 atol = 1e-12
end

@testset "define_moments! — singleton stage gets wrapped into a ChainStage" begin
    layout = StateLayout(StateAxis(:w, continuous_grid([0.0, 1.0])))
    s = IdentityStage(layout)
    mc = define_moments!(s; total = at_end(integrand = :w, reduce = sum))

    @test mc isa ChainStage
    @test !isempty(mc.spec.moments)
    @test length(mc.spec.stages) == 1

    Λ_start = [0.4, 0.6]
    Λ_end = forward!(mc, Λ_start)
    @test Λ_end == Λ_start
    @test compute_moments(mc, Λ_end, NamedTuple()).total ≈ 0.6 atol = 1e-12
end

@testset "define_moments! — re-defining a moment errors by default" begin
    layout = StateLayout(StateAxis(:w, continuous_grid([0.0, 1.0])))
    s   = IdentityStage(layout)
    mc1 = define_moments!(s; total = at_end(integrand = :w, reduce = sum))
    @test_throws ErrorException define_moments!(mc1; total = at_end(integrand = :w, reduce = sum))
end
