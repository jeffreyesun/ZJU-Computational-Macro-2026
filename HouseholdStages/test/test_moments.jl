using Test
using HouseholdStages

@testset "compute_moments — mean wealth on a uniform distribution" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0, 3.0, 4.0])),
        StateAxis(:income, [0.5, 1.5]),
    )
    chain = MarkovStage(layout; axis = :income, transition = P)
    mc = lift_moments(chain;
        avg_wealth = at_end(integrand = :wealth, reduce = sum),
    )
    buffers = allocate(mc)

    # Use a uniform distribution; one forward step keeps it uniform under P.
    Λ_start = fill(1 / 8, 4, 2)
    forward!(mc, Λ_start, buffers)
    moments = compute_moments(mc, NamedTuple())
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
    mc = lift_moments(chain;
        K = at_end(integrand = :wealth, reduce = sum),
        N = at_end(integrand = :income, reduce = sum),
    )
    buffers = allocate(mc)
    Λ_start = fill(1 / 6, 3, 2)
    forward!(mc, Λ_start, buffers)
    moments = compute_moments(mc, NamedTuple())
    @test moments.K ≈ 2.0 atol = 1e-12  # (1+2+3)/3
    @test moments.N ≈ 1.0 atol = 1e-12  # (0.5 + 1.5)/2
end

@testset "compute_moments — env-dependent integrand evaluated at call time" begin
    layout = StateLayout(StateAxis(:wealth, continuous_grid([1.0, 2.0])))
    s = IdentityStage(layout)
    mc = lift_moments(s;
        scaled_wealth = at_end(
            integrand = (cell; env) -> cell.wealth * env.scale,
            reduce    = sum,
        ),
    )
    buffers = allocate(mc)
    Λ_start = [0.5, 0.5]
    forward!(mc, Λ_start, buffers)
    moments = compute_moments(mc, (scale = 3.0,))
    # 1.0 * 0.5 * 3 + 2.0 * 0.5 * 3 = 4.5
    @test moments.scaled_wealth ≈ 4.5 atol = 1e-12
end

@testset "lift_moments — singleton stage gets wrapped into a ChainStage" begin
    layout = StateLayout(StateAxis(:w, continuous_grid([0.0, 1.0])))
    s = IdentityStage(layout)
    mc = lift_moments(s; total = at_end(integrand = :w, reduce = sum))

    @test mc isa ChainStage
    @test !isempty(mc.moments)
    @test length(mc.stages) == 1

    buffers = allocate(mc)
    Λ_start = [0.4, 0.6]
    Λ_end = forward!(mc, Λ_start, buffers)
    @test Λ_end == Λ_start
    @test compute_moments(mc, NamedTuple()).total ≈ 0.6 atol = 1e-12
end

@testset "lift_moments — re-lifting a chain that already has moments errors" begin
    layout = StateLayout(StateAxis(:w, continuous_grid([0.0, 1.0])))
    s   = IdentityStage(layout)
    mc1 = lift_moments(s; total = at_end(integrand = :w, reduce = sum))
    @test_throws ErrorException lift_moments(mc1; mean_w = at_end(integrand = :w, reduce = sum))
end
