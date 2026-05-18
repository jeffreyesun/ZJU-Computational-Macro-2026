using Test
using HouseholdStages

@testset "allocate — MarkovAlong cache=nothing, scratch non-empty" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovAlong(layout; axis = :z, transition = P)
    cache, scratch = allocate(stage, Float64)
    @test cache === nothing
    @test haskey(scratch, :perm_in) && haskey(scratch, :perm_out)
    @test size(scratch.perm_in) == size(scratch.perm_out)
end

@testset "allocate — Argmax cache exposes policy" begin
    layout = StateLayout(StateAxis(:s, categorical([:A, :B])))
    stage = Argmax(layout;
        choice_axis    = :s,
        flow_payoff    = (a; cell, env) -> (a == :B ? 1.0 : 0.0),
        next_state_idx = (cell, a) -> a == :A ? 1 : 2,
    )
    cache, scratch = allocate(stage, Float64)
    @test cache.policy === stage.policy
    @test scratch === nothing
end

@testset "allocate — LogitChoice cache is a probability tensor" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoice(layout;
        choice_axis    = :a,
        flow_payoff    = (a; cell, env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(0.5),
    )
    cache, scratch = allocate(stage, Float64)
    @test cache.choice_prob isa Array
    @test size(cache.choice_prob) == (2, 2)  # (layout dims=(2,), n_actions=2)
    @test scratch === nothing
end

@testset "allocate — ForgetfulSum has no cache or scratch" begin
    layout = StateLayout(
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:t, categorical([:a, :b, :c, :d])),
    )
    stage = ForgetfulSum(layout; forget_axis = :t)
    cache, scratch = allocate(stage, Float64)
    @test cache === nothing
    @test scratch === nothing
end

@testset "allocate — StageChain returns per-stage tuples" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s1 = MarkovAlong(layout; axis = :z, transition = P)
    s2 = MarkovAlong(layout; axis = :z, transition = P)
    chain = s1 ∘ₛ s2
    caches, scratches = allocate(chain, Float64)
    @test length(caches) == 2
    @test length(scratches) == 2
    @test caches[1] === nothing
    @test caches[2] === nothing
    @test haskey(scratches[1], :perm_in)
end

@testset "explicit (cache, scratch) backward/forward call" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovAlong(layout; axis = :z, transition = P)
    cache, scratch = allocate(stage, Float64)
    V_end = ones(2, 3)
    V_start = backward!(stage, V_end, nothing, cache, scratch)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    Λ_start = rand(2, 3); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, cache, scratch, nothing)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end
