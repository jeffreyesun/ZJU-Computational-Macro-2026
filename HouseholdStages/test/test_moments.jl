using Test
using HouseholdStages

@testset "compute_moments — mean wealth on a uniform distribution" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0, 3.0, 4.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    chain = MarkovAlong(layout; axis = :income, transition = P)
    mc = lift_moments(chain;
        avg_wealth = at_end(integrand = :wealth, reduce = sum),
    )
    caches, scratches = allocate(mc)

    # Use a uniform distribution; one forward step keeps it uniform under P.
    Λ_start = fill(1 / 8, 4, 2)
    forward!(mc, Λ_start, caches, scratches)
    moments = compute_moments(mc, NamedTuple())
    # Σ_w wealth * Σ_z (1/8) = (1+2+3+4) / 4 = 2.5
    @test moments.avg_wealth ≈ 2.5 atol = 1e-12
end

@testset "compute_moments — multiple specs in one call" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    chain = MarkovAlong(layout; axis = :income, transition = P)
    mc = lift_moments(chain;
        K = at_end(integrand = :wealth, reduce = sum),
        N = at_end(integrand = :income, reduce = sum),
    )
    caches, scratches = allocate(mc)
    Λ_start = fill(1 / 6, 3, 2)
    forward!(mc, Λ_start, caches, scratches)
    moments = compute_moments(mc, NamedTuple())
    @test moments.K ≈ 2.0 atol = 1e-12  # (1+2+3)/3
    @test moments.N ≈ 1.0 atol = 1e-12  # (0.5 + 1.5)/2
end

@testset "compute_moments — env-dependent integrand picked up in env_slice" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0])),
    )
    s = IdentityStage(layout)
    mc = lift_moments(s;
        scaled_wealth = at_end(
            integrand    = (cell; env) -> cell.wealth * env.scale,
            reduce       = sum,
            closure_deps = (:scale,),
        ),
    )
    @test :scale in effective_env_slice(mc)
end

@testset "compute_moments — MomentedChain delegates stage methods to inner" begin
    layout = StateLayout(StateAxis(:w, continuous_grid([0.0, 1.0])))
    s = IdentityStage(layout)
    mc = lift_moments(s; total = at_end(integrand = :w, reduce = sum))

    @test allocate(mc) === (nothing, nothing)
    Λ_start = [0.4, 0.6]
    cache, scratch = allocate(mc)
    Λ_end = forward!(mc, Λ_start, cache, scratch)
    @test Λ_end == Λ_start
end
