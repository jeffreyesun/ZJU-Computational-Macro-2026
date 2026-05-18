using Test
using HouseholdStages
using LinearAlgebra

@testset "HouseholdStages" begin

include("test_layout.jl")
include("test_param.jl")
include("test_payoff_fn.jl")
include("test_workspace.jl")
include("test_markov_along.jl")
include("test_discrete_choice.jl")
include("test_migration.jl")
include("test_monotone_argmax.jl")
include("test_identity_stage.jl")
include("test_utility_stage.jl")
include("test_env_deps.jl")
include("test_moments.jl")
include("test_product.jl")
include("test_asset_price_change.jl")
include("test_borrowing_constraint.jl")
include("test_lift_jacobian.jl")
include("test_sequence_space.jl")
include("test_outer_loop.jl")

@testset "ForgetfulSum — drops one axis, 3D → 2D" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.0, 1.5])),
        StateAxis(:taste,  categorical([:a, :b, :c, :d, :e])),
    )
    stage = ForgetfulSum(layout; forget_axis = :taste)
    @test layout_size(stage.output_layout) == (4, 3)
    @test size(stage.V_start) == (4, 3, 5)
    @test size(stage.Λ_end)   == (4, 3)

    kernel, scratch = allocate(stage)
    V_end = reshape(Float64.(1:12), (4, 3))
    V_start = backward!(stage, V_end, nothing, kernel, scratch)
    @test size(V_start) == (4, 3, 5)
    for t in 1:5
        @test V_start[:, :, t] == V_end
    end

    Λ_start = rand(Float64, 4, 3, 5); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, kernel, scratch)
    @test size(Λ_end) == (4, 3)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
    expected = dropdims(sum(Λ_start; dims = 3); dims = 3)
    @test all(isapprox.(Λ_end, expected; atol = 1e-14))
end

@testset "ForgetfulSum — drops a middle axis, 4D → 3D" begin
    layout = StateLayout(
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z, discrete_finite([0.5, 1.0, 1.5, 2.0])),
        StateAxis(:transient, categorical([:lo, :hi])),
        StateAxis(:loc, categorical([:A, :B, :C, :D, :E])),
    )
    stage = ForgetfulSum(layout; forget_axis = :transient)
    @test layout_size(stage.output_layout) == (3, 4, 5)

    kernel, scratch = allocate(stage)
    V_end = randn(3, 4, 5)
    V_start = backward!(stage, V_end, nothing, kernel, scratch)
    for w in 1:3, z in 1:4, t in 1:2, loc in 1:5
        @test V_start[w, z, t, loc] == V_end[w, z, loc]
    end

    Λ_start = rand(3, 4, 2, 5); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, kernel, scratch)
    @test isapprox(sum(Λ_end), sum(Λ_start); atol = 1e-12)
    expected = dropdims(sum(Λ_start; dims = 3); dims = 3)
    @test all(isapprox.(Λ_end, expected; atol = 1e-14))
end

@testset "ForgetfulSum — duality identity" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 0.5, 1.0, 1.5])),
        StateAxis(:income, discrete_finite([0.5, 1.0, 1.5])),
        StateAxis(:taste, categorical([:a, :b, :c, :d, :e])),
    )
    stage = ForgetfulSum(layout; forget_axis = :income)

    kernel, scratch = allocate(stage)
    V_out  = randn(4, 5)
    Λ_in   = rand(4, 3, 5); Λ_in ./= sum(Λ_in)

    V_in  = backward!(stage, V_out, nothing, kernel, scratch)
    Λ_out = forward!(stage, Λ_in, kernel, scratch)

    @test isapprox(sum(V_in .* Λ_in), sum(V_out .* Λ_out); atol = 1e-12)
end

@testset "StageChain — composition is associative and length-2 works" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s1 = MarkovAlong(layout; axis = :z, transition = P)
    s2 = MarkovAlong(layout; axis = :z, transition = P)
    chain = s1 ∘ₛ s2
    @test chain isa StageChain
    @test length(chain.stages) == 2

    chain3 = (s1 ∘ₛ s2) ∘ₛ s1
    chain3b = s1 ∘ₛ (s2 ∘ₛ s1)
    @test length(chain3.stages) == length(chain3b.stages) == 3

    # End-to-end: ones should remain ones after forward passes through the chain.
    Λ = ones(2) ./ 2
    caches, scratches = allocate(chain, Float64)
    Λ_end = forward!(chain, copy(Λ), caches, scratches)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end

end
