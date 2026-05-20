using Test
using HouseholdStages

@testset "kernel cache — hit on matching (V_end, env)" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :z, transition = P)

    V_end = randn(3, 2)
    env   = (;)
    backward!(stage, V_end, env)

    @test stage.buffer.cache.kernel_valid
    @test stage.buffer.cache.last_V_hash == hash(V_end)
    @test stage.buffer.cache.last_env === env

    # forward! with the matching (V_end, env) hits the cache.
    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    @test forward!(stage, Λ_start, V_end, env) isa AbstractArray
end

@testset "kernel cache — error on stale (V_end, env)" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    stage  = MarkovStage(layout; axis = :z, transition = P)

    V_end_seed = randn(2)
    backward!(stage, V_end_seed, (;))

    # A different V_end is stale.
    V_end_stale = V_end_seed .+ 1.0
    Λ_start = [0.5, 0.5]
    @test_throws ErrorException forward!(stage, Λ_start, V_end_stale, (;))
end

@testset "kernel cache — reseat_if_stale=true re-runs backward" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    stage  = MarkovStage(layout; axis = :z, transition = P)

    # First backward at V_end_a; cache holds (V_end_a, env).
    V_end_a = randn(2)
    backward!(stage, V_end_a, (;))

    # forward! with V_end_b should reseat.
    V_end_b = V_end_a .+ 0.5
    Λ_start = [0.5, 0.5]
    Λ_end   = forward!(stage, Λ_start, V_end_b, (;); reseat_if_stale = true)
    @test Λ_end isa AbstractArray
    @test stage.buffer.cache.last_V_hash == hash(V_end_b)
end

@testset "kernel cache — explicit invalidate!" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    stage  = MarkovStage(layout; axis = :z, transition = P)

    V_end = randn(2)
    backward!(stage, V_end, (;))
    @test stage.buffer.cache.kernel_valid

    invalidate!(stage)
    @test !stage.buffer.cache.kernel_valid

    # forward! with check=true on an explicitly-invalidated cache errors.
    @test_throws ErrorException forward!(stage, [0.5, 0.5], V_end, (;))

    # check=false bypasses entirely.
    @test forward!(stage, [0.5, 0.5], V_end, (;); check = false) isa AbstractArray
end

@testset "kernel cache — chain-level cache via terminal stage" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s1 = MarkovStage(layout; axis = :z, transition = P)
    s2 = MarkovStage(layout; axis = :z, transition = P)
    chain = s1 ∘ s2

    V_end = randn(2)
    backward!(chain, V_end, (;))

    @test chain.buffer.cache.kernel_valid
    @test chain.buffer.cache.last_V_hash == hash(V_end)

    # Stale (V_end) on chain forward errors.
    V_end_stale = V_end .+ 0.5
    @test_throws ErrorException forward!(chain, [0.5, 0.5], V_end_stale, (;))

    # reseat_if_stale=true succeeds.
    Λ_end = forward!(chain, [0.5, 0.5], V_end_stale, (;); reseat_if_stale = true)
    @test Λ_end isa AbstractArray
end

@testset "kernel cache — in-place V mutation is detected" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    stage  = MarkovStage(layout; axis = :z, transition = P)

    V_end = randn(2)
    backward!(stage, V_end, (;))

    # In-place modification: same identity, different hash.
    V_end .*= 0.5

    # The cache hashes the full array, so the mutation is caught.
    @test_throws ErrorException forward!(stage, [0.5, 0.5], V_end, (;))
end
