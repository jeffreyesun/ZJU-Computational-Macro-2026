using Test
using HouseholdStages

@testset "allocate — MarkovStage kernel=nothing, scratch non-empty" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z, discrete_finite([0.5, 1.5])),
    )
    stage = MarkovStage(layout; axis = :z, transition = P)
    buffers = allocate(stage, Float64)
    @test buffers.kernel === nothing
    @test haskey(buffers.scratch, :perm_in) && haskey(buffers.scratch, :perm_out)
    @test size(buffers.scratch.perm_in) == size(buffers.scratch.perm_out)
end

@testset "allocate — ArgmaxStage kernel exposes policy" begin
    layout = StateLayout(StateAxis(:s, categorical([:A, :B])))
    stage = ArgmaxStage(layout;
        choice_axis    = :s,
        flow_payoff    = (a; cell, env) -> (a == :B ? 1.0 : 0.0),
        next_state_idx = (cell, a) -> a == :A ? 1 : 2,
    )
    buffers = allocate(stage, Float64)
    @test buffers.kernel.policy === stage.policy
    @test buffers.scratch === nothing
end

@testset "allocate — LogitChoiceStage kernel is a probability tensor" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (a; cell, env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(0.5),
    )
    buffers = allocate(stage, Float64)
    @test buffers.kernel.choice_prob isa Array
    @test size(buffers.kernel.choice_prob) == (2, 2)  # (layout dims=(2,), n_actions=2)
    @test buffers.scratch === nothing
end

@testset "allocate — ForgetfulSumStage has no kernel or scratch" begin
    layout = StateLayout(
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:t, categorical([:a, :b, :c, :d])),
    )
    stage = ForgetfulSumStage(layout; forget_axis = :t)
    buffers = allocate(stage, Float64)
    @test buffers.kernel === nothing
    @test buffers.scratch === nothing
end

@testset "allocate — ChainStage returns per-stage tuple" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s1 = MarkovStage(layout; axis = :z, transition = P)
    s2 = MarkovStage(layout; axis = :z, transition = P)
    chain = s1 ∘ₛ s2
    buffers = allocate(chain, Float64)
    @test buffers isa Tuple
    @test length(buffers) == 2
    @test buffers[1].kernel === nothing
    @test buffers[2].kernel === nothing
    @test haskey(buffers[1].scratch, :perm_in)
    @test haskey(buffers[2].scratch, :perm_in)
end

@testset "single-stage backward/forward via buffers" begin
    P = [0.9 0.1; 0.2 0.8]
    layout = StateLayout(
        StateAxis(:z, discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    stage = MarkovStage(layout; axis = :z, transition = P)
    buffers = allocate(stage, Float64)
    V_end = ones(2, 3)
    V_start = backward!(stage, V_end, nothing, buffers)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    Λ_start = rand(2, 3); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, buffers, nothing)
    @test isapprox(sum(Λ_end), 1.0; atol = 1e-12)
end
