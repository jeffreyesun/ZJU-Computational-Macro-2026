using Test
using HouseholdStages
using LinearAlgebra

@testset "MarkovStage — construction and field checks" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :income, transition = P)
    @test stage isa MarkovStage
    @test stage.axis === :income
    @test stage.axis_dim == 2
    @test stage.transition === P
    @test stage.input_layout === layout
    @test stage.output_layout === layout
    @test size(stage.V_start) == (4, 2)
    @test size(stage.Λ_end)   == (4, 2)
end

@testset "MarkovStage — backward & forward correctness" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :income, transition = P)
    buffers = allocate(stage, Float64)

    V_end = ones(4, 2)
    V_start = backward!(stage, V_end, nothing, buffers)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    V_end2 = zeros(4, 2); V_end2[:, 1] .= 1.0; V_end2[:, 2] .= 4.0
    V_start2 = backward!(stage, V_end2, nothing, buffers)
    @test all(isapprox.(V_start2[:, 1], 0.7*1.0 + 0.3*4.0; atol = 1e-12))
    @test all(isapprox.(V_start2[:, 2], 0.3*1.0 + 0.7*4.0; atol = 1e-12))

    Λ_start = rand(4, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, buffers, nothing)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end

@testset "MarkovStage — duality identity" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 0.5, 1.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :z, transition = P)
    buffers = allocate(stage)

    V_out = randn(3, 2)
    Λ_in  = rand(3, 2); Λ_in ./= sum(Λ_in)

    V_in  = backward!(stage, V_out, nothing, buffers)
    Λ_out = forward!(stage, Λ_in, buffers, nothing)

    # For a pure-Markov stage the flow payoff `r` is zero, so duality
    # reduces to ⟨V_in, Λ_in⟩ ≈ ⟨V_out, Λ_out⟩.
    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out); atol = 1e-12)
end

@testset "MarkovStage — backward axis=1 (first dim)" begin
    # axis_dim == 1 hits the no-permute fast path in _markov_apply!.
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovStage(layout; axis = :z, transition = P)
    buffers = allocate(stage)
    V_end = ones(2, 3)
    V_start = backward!(stage, V_end, nothing, buffers)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))
end

@testset "MarkovStage — static_env_deps is empty" begin
    @test static_env_deps(MarkovStage) === NamedTuple()
end

@testset "MarkovStage — type stability" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovStage(layout; axis = :z, transition = P)
    buffers = allocate(stage)
    V_end = ones(2, 3)
    @inferred backward!(stage, V_end, nothing, buffers)
end
