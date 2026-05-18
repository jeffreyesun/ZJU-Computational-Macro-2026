using Test
using HouseholdStages

@testset "static_env_deps defaults to empty NamedTuple" begin
    @test static_env_deps(MarkovAlong) === NamedTuple()
    @test static_env_deps(Argmax) === NamedTuple()
    @test static_env_deps(LogitChoice) === NamedTuple()
    @test static_env_deps(ForgetfulSum) === NamedTuple()
    @test static_env_deps(IdentityStage) === NamedTuple()
end

@testset "effective_env_slice picks up closure_deps" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = Argmax(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> (a == 1 ? 0.0 : env.bonus),
        next_state_idx = (cell, a) -> a,
        closure_deps   = (:bonus,),
    )
    @test :bonus in effective_env_slice(stage)
end

@testset "effective_env_slice picks up swept Param" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoice(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
    )
    @test :ξ in effective_env_slice(stage)
end

@testset "validate_env catches missing fields" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoice(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
    )
    @test_throws ErrorException validate_env(stage, NamedTuple())
    @test validate_env(stage, (ξ = 0.5,)) === nothing
end

@testset "chain_env_names merges per-stage slices" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    s1 = MarkovAlong(layout; axis = :a, transition = P)
    s2 = LogitChoice(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> (a == 1 ? 0.0 : env.bonus),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
        closure_deps   = (:bonus,),
    )
    chain = s1 ∘ₛ s2
    names = chain_env_names(chain)
    @test :ξ in names
    @test :bonus in names
end

@testset "Param mode-flip updates effective slice" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoice(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(0.5),       # calibrated; nothing in slice
    )
    @test isempty(effective_env_slice(stage))
    stage.ε.val = :ξ
    @test :ξ in effective_env_slice(stage)
end
