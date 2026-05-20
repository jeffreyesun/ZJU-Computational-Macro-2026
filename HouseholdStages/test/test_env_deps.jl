using Test
using HouseholdStages

@testset "static_env_deps defaults to empty NamedTuple" begin
    @test static_env_deps(HouseholdStages.MarkovStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.ArgmaxStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.LogitChoiceStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.ForgetfulSumStageSpec) === NamedTuple()
    @test static_env_deps(HouseholdStages.IdentityStageSpec) === NamedTuple()
end

@testset "effective_env_slice is empty when closures aren't introspected" begin
    # User closures read `env.bonus`, but the package no longer requires
    # `closure_deps` to be declared. effective_env_slice reflects only
    # static_env_deps + swept Param keys, so a stage whose env reads
    # flow exclusively through closures has an empty slice.
    layout = StateLayout(StateAxis(:a, [1, 2]))
    stage = ArgmaxStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> (a == 1 ? 0.0 : env.bonus),
        next_state_idx = (cell, a) -> a,
    )
    @test isempty(effective_env_slice(stage))
end

@testset "effective_env_slice picks up swept Param" begin
    layout = StateLayout(StateAxis(:a, [1, 2]))
    stage = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
    )
    @test :ξ in effective_env_slice(stage)
end

@testset "validate_env catches missing sweep keys" begin
    layout = StateLayout(StateAxis(:a, [1, 2]))
    stage = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
    )
    @test_throws ErrorException validate_env(stage, NamedTuple())
    @test validate_env(stage, (ξ = 0.5,)) === nothing
end

@testset "chain_env_names merges per-stage swept-Param slices" begin
    P = [0.6 0.4; 0.25 0.75]
    layout = StateLayout(StateAxis(:a, [1, 2]))
    s1 = MarkovStage(layout; axis = :a, transition = P)
    s2 = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(:ξ),
    )
    chain = s1 ∘ s2
    @test :ξ in chain_env_names(chain)
end

@testset "Param mode-flip updates effective slice" begin
    layout = StateLayout(StateAxis(:a, [1, 2]))
    stage = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(0.5),       # calibrated; nothing in slice
    )
    @test isempty(effective_env_slice(stage))
    stage.spec.ε.val = :ξ
    @test :ξ in effective_env_slice(stage)
end
