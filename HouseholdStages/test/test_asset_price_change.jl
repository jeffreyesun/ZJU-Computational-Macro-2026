using Test
using HouseholdStages

@testset "AssetPriceChange — constructor returns a WealthChange" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:h,      discrete_finite([0.0, 1.0, 2.0])),
    )
    stage = AssetPriceChange(layout; holdings_axis = :h)
    @test stage isa WealthChange
    @test stage.wealth_axis === :wealth
    @test :q in effective_env_slice(stage)
    @test :q_last in effective_env_slice(stage)
end

@testset "AssetPriceChange — wealth_post closure matches hand-built recipe" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0, 4.0])),
        StateAxis(:h,      discrete_finite([0.0, 1.0, 2.0])),
    )
    stage_sugar = AssetPriceChange(layout; holdings_axis = :h)
    # Hand-built closure: WealthChange passes env as a Ref, so the closure unwraps with env[].
    stage_hand  = WealthChange(layout;
        wealth_post  = (cell; env) -> begin
            e = env[]
            return cell.wealth + (e.q - e.q_last) * cell.h
        end,
        wealth_axis  = :wealth,
        closure_deps = (:q, :q_last),
    )

    env = (; q = 1.10, q_last = 1.00)
    k_s, sc_s = allocate(stage_sugar)
    k_h, sc_h = allocate(stage_hand)

    # Backward: feed a non-trivial V_end and check identical V_start.
    V_end = reshape(Float64.(1:15), (5, 3))
    V_s = copy(backward!(stage_sugar, V_end, env, k_s, sc_s))
    V_h = copy(backward!(stage_hand,  V_end, env, k_h, sc_h))
    @test all(isapprox.(V_s, V_h; atol = 1e-12))

    # Forward: check distribution-push identity.
    Λ_start = rand(5, 3); Λ_start ./= sum(Λ_start)
    Λ_s = copy(forward!(stage_sugar, Λ_start, k_s, sc_s))
    Λ_h = copy(forward!(stage_hand,  Λ_start, k_h, sc_h))
    @test all(isapprox.(Λ_s, Λ_h; atol = 1e-12))
end

@testset "AssetPriceChange — q == q_last is identity (up to interpolation)" begin
    # With no price revaluation, the stage should leave V untouched.
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:h,      discrete_finite([0.0, 1.0])),
    )
    stage = AssetPriceChange(layout; holdings_axis = :h)
    env   = (; q = 1.0, q_last = 1.0)

    kernel, scratch = allocate(stage)
    V_end = reshape(Float64.(1:8), (4, 2))
    V_start = backward!(stage, V_end, env, kernel, scratch)
    @test all(isapprox.(V_start, V_end; atol = 1e-12))

    Λ_start = rand(4, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, kernel, scratch)
    @test all(isapprox.(Λ_end, Λ_start; atol = 1e-12))
end

@testset "AssetPriceChange — custom field names" begin
    # Verify the q_field / q_last_field / wealth_axis / holdings_axis kwargs work.
    layout = StateLayout(
        StateAxis(:b, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:k, discrete_finite([0.0, 1.0, 2.0])),
    )
    stage = AssetPriceChange(layout;
        holdings_axis = :k,
        wealth_axis   = :b,
        q_field       = :p_now,
        q_last_field  = :p_prev,
    )
    @test stage.wealth_axis === :b
    @test :p_now in effective_env_slice(stage)
    @test :p_prev in effective_env_slice(stage)

    env = (; p_now = 1.5, p_prev = 1.0)
    kernel, scratch = allocate(stage)
    V_end = reshape(Float64.(1:12), (4, 3))
    @test backward!(stage, V_end, env, kernel, scratch) isa AbstractArray
end
