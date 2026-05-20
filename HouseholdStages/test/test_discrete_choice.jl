using Test
using HouseholdStages

@testset "ArgmaxStage — 2-state re-choose, action 2 always preferred (+1)" begin
    layout = StateLayout(StateAxis(:s, categorical([:A, :B])))
    stage = ArgmaxStage(layout;
        choice_axis = :s,
        flow_payoff = (cell, a; env) -> (a == :B ? 1.0 : 0.0),
        next_state_idx = (cell, a) -> a == :A ? 1 : 2,
    )

    V_end = Float64[0.0, 0.0]
    V_start = backward!(stage, V_end, nothing)
    @test V_start == [1.0, 1.0]
    @test stage.buffer.kernel.policy == [2, 2]

    Λ_start = Float64[1.0, 0.0]
    Λ_end = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ 1.0
    @test Λ_end == [0.0, 1.0]
end

@testset "ArgmaxStage — -Inf payoff is skipped (unavailable action)" begin
    layout = StateLayout(StateAxis(:s, discrete_finite([1, 2])))
    stage = ArgmaxStage(layout;
        choice_axis = :s,
        flow_payoff = (cell, a; env) -> (a == 2 && cell.s == 1) ? -Inf : 0.0,
        next_state_idx = (cell, a) -> a,
    )
    V_end = zeros(2)
    V_start = backward!(stage, V_end, nothing)
    @test V_start == [0.0, 0.0]
    @test stage.buffer.kernel.policy[1] == 1
end

@testset "LogitChoiceStage — 2 actions = 2 choice-axis levels, softmax probs" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    ε = 0.5
    stage = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> (a == 1 ? 0.0 : 1.0),
        next_state_idx = (cell, a) -> a,
        ε              = Param(ε),
    )
    V_end = Float64[0.0, 0.0]
    V_start = backward!(stage, V_end, nothing)
    expected = 1.0 + ε * log(1 + exp(-1/ε))
    @test V_start[1] ≈ expected
    @test V_start[2] ≈ expected

    p_row1 = stage.buffer.kernel.choice_prob[1, :]
    @test sum(p_row1) ≈ 1.0
    @test p_row1[2] > p_row1[1]

    Λ_start = Float64[1.0, 0.0]
    Λ_end = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ 1.0
    @test Λ_end[1] ≈ p_row1[1]
    @test Λ_end[2] ≈ p_row1[2]
end

@testset "LogitChoiceStage — Param swept via env" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> (a == 1 ? 0.0 : 1.0),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
    )
    V_end = Float64[0.0, 0.0]
    V_start_sharp  = copy(backward!(stage, V_end, (ξ = 0.01,)))
    V_start_smooth = copy(backward!(stage, V_end, (ξ = 1.0,)))
    @test V_start_sharp[1] < V_start_smooth[1]
    @test V_start_sharp[1] ≈ 1.0 atol = 0.05
end

@testset "ArgmaxStage / LogitChoiceStage — static_env_deps" begin
    @test static_env_deps(HouseholdStages.ArgmaxStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.LogitChoiceStageSpec) === NamedTuple()
end
