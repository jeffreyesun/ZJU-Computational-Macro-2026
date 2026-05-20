using Test
using HouseholdStages
using ForwardDiff: ForwardDiff, Dual

@testset "UtilityStage — backward adds u(s) to V_end" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:income, [0.5, 1.0]),
    )
    u = (cell; env) -> cell.wealth + env.bonus
    stage = UtilityStage(layout; utility = u)

    V_end = zeros(3, 2)
    env = (bonus = 10.0,)
    V_start = backward!(stage, V_end, env)

    # u(s) = cell.wealth + 10 (since V_end = 0).
    @test V_start[1, 1] ≈ 10.0     # wealth = 0
    @test V_start[2, 1] ≈ 11.0     # wealth = 1
    @test V_start[3, 1] ≈ 12.0     # wealth = 2
    @test V_start[1, 2] ≈ 10.0     # wealth = 0 (income axis doesn't enter u)
end

@testset "UtilityStage — backward composes additively with V_end" begin
    layout = StateLayout(StateAxis(:z, [1, 2, 3]))
    u = (cell; env) -> Float64(cell.z)
    stage = UtilityStage(layout; utility = u)

    V_end = [10.0, 20.0, 30.0]
    V_start = backward!(stage, V_end, NamedTuple())
    @test V_start ≈ [11.0, 22.0, 33.0]
end

@testset "UtilityStage — forward is identity on Λ" begin
    layout = StateLayout(
        StateAxis(:w, continuous_grid([0.0, 0.5, 1.0])),
        StateAxis(:z, [:a, :b]),
    )
    u = (cell; env) -> 1.0   # any utility; forward shouldn't read it
    stage = UtilityStage(layout; utility = u)

    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start)
    @test Λ_end == Λ_start
    @test Λ_end !== Λ_start    # written into the stage's buffer
end

@testset "UtilityStage — duality identity (with flow payoff r = u)" begin
    # ⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩
    # With Λ_out = Λ_in and r = u, this reduces to V_in = V_out + u.
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, [0.5, 1.0, 1.5]),
    )
    u = (cell; env) -> log1p(cell.wealth) * cell.income
    stage = UtilityStage(layout; utility = u)

    V_out = randn(4, 3)
    Λ_in  = rand(4, 3); Λ_in ./= sum(Λ_in)

    V_in  = copy(backward!(stage, V_out, NamedTuple()))
    Λ_out = copy(forward!(stage,  Λ_in))

    r = V_in .- V_out
    @test isapprox(sum(V_in .* Λ_in),
                   sum(V_out .* Λ_out) + sum(r .* Λ_in);
                   atol = 1e-12)
end

@testset "UtilityStage — effective_env_slice is empty (no introspection)" begin
    layout = StateLayout(StateAxis(:z, [1, 2]))
    u = (cell; env) -> env.σ * cell.z
    stage = UtilityStage(layout; utility = u)
    @test effective_env_slice(stage) == ()

    stage_nodeps = UtilityStage(layout; utility = (cell; env) -> 0.0)
    @test effective_env_slice(stage_nodeps) == ()
end

@testset "UtilityStage — composition with MarkovStage" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(StateAxis(:z, [0.5, 1.5]))
    markov = MarkovStage(layout; axis = :z, transition = P)
    util   = UtilityStage(layout; utility = (cell; env) -> cell.z)

    chain = markov ∘ util
    V_start = backward!(chain, zeros(2), NamedTuple())

    @test V_start ≈ P * [0.5, 1.5]   # = [0.8, 1.2]
end

@testset "UtilityStage — Dual-typed buffer rebuild for AD" begin
    layout = StateLayout(StateAxis(:z, [1, 2, 3]))
    u = (cell; env) -> Float64(cell.z) * env.a
    stage = UtilityStage(layout; utility = u)

    DualT = ForwardDiff.Dual{Nothing, Float64, 1}
    stage_dual = with_eltype(stage, DualT)
    @test eltype(stage_dual.buffer.V_start) <: ForwardDiff.Dual
    @test eltype(stage_dual.buffer.Λ_end)   <: ForwardDiff.Dual

    V_end_dual = zeros(DualT, 3)
    env_dual   = (a = DualT(2.0, ForwardDiff.Partials((1.0,))),)
    V_start    = backward!(stage_dual, V_end_dual, env_dual)
    @test [ForwardDiff.partials(V_start[i], 1) for i in 1:3] == [1.0, 2.0, 3.0]
end

@testset "UtilityStage — static_env_deps" begin
    @test static_env_deps(HouseholdStages.UtilityStageSpec) === NamedTuple()
end
