using Test
using HouseholdStages

@testset "Argmax — 2-state re-choose, action 2 always preferred (+1)" begin
    layout = StateLayout(StateAxis(:s, categorical([:A, :B])))
    stage = Argmax(layout;
        choice_axis = :s,
        flow_payoff = (cell, a; env) -> (a == :B ? 1.0 : 0.0),
        next_state_idx = (cell, a) -> a == :A ? 1 : 2,
    )

    V_end = Float64[0.0, 0.0]
    cache, scratch = allocate(stage)
    V_start = backward!(stage, V_end, nothing, cache, scratch)
    @test V_start == [1.0, 1.0]
    @test stage.policy == [2, 2]

    Λ_start = Float64[1.0, 0.0]
    Λ_end = forward!(stage, Λ_start, cache, scratch, nothing)
    @test sum(Λ_end) ≈ 1.0
    @test Λ_end == [0.0, 1.0]
end

@testset "Argmax — -Inf payoff is skipped (unavailable action)" begin
    layout = StateLayout(StateAxis(:s, discrete_finite([1, 2])))
    stage = Argmax(layout;
        choice_axis = :s,
        flow_payoff = (cell, a; env) -> (a == 2 && cell.s == 1) ? -Inf : 0.0,
        next_state_idx = (cell, a) -> a,
    )
    V_end = zeros(2)
    cache, scratch = allocate(stage)
    V_start = backward!(stage, V_end, nothing, cache, scratch)
    @test V_start == [0.0, 0.0]
    @test stage.policy[1] == 1
end

@testset "LogitChoice — 2 actions = 2 choice-axis levels, softmax probs" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    ε = 0.5
    stage = LogitChoice(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> (a == 1 ? 0.0 : 1.0),
        next_state_idx = (cell, a) -> a,
        ε              = Param(ε),
    )
    cache, scratch = allocate(stage)
    V_end = Float64[0.0, 0.0]
    V_start = backward!(stage, V_end, nothing, cache, scratch)
    expected = 1.0 + ε * log(1 + exp(-1/ε))
    @test V_start[1] ≈ expected
    @test V_start[2] ≈ expected

    p_row1 = cache.choice_prob[1, :]
    @test sum(p_row1) ≈ 1.0
    @test p_row1[2] > p_row1[1]

    Λ_start = Float64[1.0, 0.0]
    Λ_end = forward!(stage, Λ_start, cache, scratch, nothing)
    @test sum(Λ_end) ≈ 1.0
    @test Λ_end[1] ≈ p_row1[1]
    @test Λ_end[2] ≈ p_row1[2]
end

@testset "LogitChoice — Param swept via env" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoice(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> (a == 1 ? 0.0 : 1.0),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
    )
    cache, scratch = allocate(stage)
    V_end = Float64[0.0, 0.0]
    V_start_sharp  = copy(backward!(stage, V_end, (ξ = 0.01,), cache, scratch))
    V_start_smooth = copy(backward!(stage, V_end, (ξ = 1.0,),  cache, scratch))
    @test V_start_sharp[1] < V_start_smooth[1]
    @test V_start_sharp[1] ≈ 1.0 atol = 0.05
end

@testset "Argmax / LogitChoice — static_env_deps" begin
    @test static_env_deps(Argmax) === NamedTuple()
    @test static_env_deps(LogitChoice) === NamedTuple()
end
