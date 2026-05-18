using Test
using HouseholdStages
using LinearAlgebra

@testset "MarkovAlong — construction and field checks" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovAlong(layout; axis = :income, transition = P)
    @test stage isa MarkovAlong
    @test stage.axis === :income
    @test stage.axis_dim == 2
    @test stage.transition === P
    @test stage.input_layout === layout
    @test stage.output_layout === layout
    @test size(stage.V_start) == (4, 2)
    @test size(stage.Λ_end)   == (4, 2)
end

@testset "MarkovAlong — backward & forward correctness" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovAlong(layout; axis = :income, transition = P)
    cache, scratch = allocate(stage, Float64)

    V_end = ones(4, 2)
    V_start = backward!(stage, V_end, nothing, cache, scratch)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    V_end2 = zeros(4, 2); V_end2[:, 1] .= 1.0; V_end2[:, 2] .= 4.0
    V_start2 = backward!(stage, V_end2, nothing, cache, scratch)
    @test all(isapprox.(V_start2[:, 1], 0.7*1.0 + 0.3*4.0; atol = 1e-12))
    @test all(isapprox.(V_start2[:, 2], 0.3*1.0 + 0.7*4.0; atol = 1e-12))

    Λ_start = rand(4, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, cache, scratch, nothing)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end

@testset "MarkovAlong — duality identity" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 0.5, 1.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    stage = MarkovAlong(layout; axis = :z, transition = P)
    cache, scratch = allocate(stage)

    V_out = randn(3, 2)
    Λ_in  = rand(3, 2); Λ_in ./= sum(Λ_in)

    V_in  = backward!(stage, V_out, nothing, cache, scratch)
    Λ_out = forward!(stage, Λ_in, cache, scratch, nothing)

    # For a pure-Markov stage the flow payoff `r` is zero, so duality
    # reduces to ⟨V_in, Λ_in⟩ ≈ ⟨V_out, Λ_out⟩.
    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out); atol = 1e-12)
end

@testset "MarkovAlong — backward axis=1 (first dim)" begin
    # axis_dim == 1 hits the no-permute fast path in _markov_apply!.
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovAlong(layout; axis = :z, transition = P)
    cache, scratch = allocate(stage)
    V_end = ones(2, 3)
    V_start = backward!(stage, V_end, nothing, cache, scratch)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))
end

@testset "MarkovAlong — static_env_deps is empty" begin
    @test static_env_deps(MarkovAlong) === NamedTuple()
end

@testset "MarkovAlong — type stability" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovAlong(layout; axis = :z, transition = P)
    cache, scratch = allocate(stage)
    V_end = ones(2, 3)
    @inferred backward!(stage, V_end, nothing, cache, scratch)
end
