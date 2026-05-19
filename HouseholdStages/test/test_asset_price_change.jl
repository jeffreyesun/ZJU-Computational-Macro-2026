using Test
using HouseholdStages

@testset "AssetPriceChangeStage — constructor returns a WealthChangeStage" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:h,      [0.0, 1.0, 2.0]),
    )
    stage = AssetPriceChangeStage(layout; holdings_axis = :h)
    @test stage isa WealthChangeStage
    @test stage.wealth_axis === :wealth
end

@testset "AssetPriceChangeStage — wealth_post closure matches hand-built recipe" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0, 4.0])),
        StateAxis(:h,      [0.0, 1.0, 2.0]),
    )
    stage_sugar = AssetPriceChangeStage(layout; holdings_axis = :h)
    stage_hand  = WealthChangeStage(layout;
        wealth_post = (cell; env) -> cell.wealth + (env.q - env.q_last) * cell.h,
        wealth_axis = :wealth,
    )

    env = (; q = 1.10, q_last = 1.00)
    buf_s = allocate(stage_sugar)
    buf_h = allocate(stage_hand)

    V_end = reshape(Float64.(1:15), (5, 3))
    V_s = copy(backward!(stage_sugar, V_end, env, buf_s))
    V_h = copy(backward!(stage_hand,  V_end, env, buf_h))
    @test all(isapprox.(V_s, V_h; atol = 1e-12))

    Λ_start = rand(5, 3); Λ_start ./= sum(Λ_start)
    Λ_s = copy(forward!(stage_sugar, Λ_start, buf_s))
    Λ_h = copy(forward!(stage_hand,  Λ_start, buf_h))
    @test all(isapprox.(Λ_s, Λ_h; atol = 1e-12))
end

@testset "AssetPriceChangeStage — q == q_last is identity (up to interpolation)" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:h,      [0.0, 1.0]),
    )
    stage = AssetPriceChangeStage(layout; holdings_axis = :h)
    env   = (; q = 1.0, q_last = 1.0)

    buffers = allocate(stage)
    V_end = reshape(Float64.(1:8), (4, 2))
    V_start = backward!(stage, V_end, env, buffers)
    @test all(isapprox.(V_start, V_end; atol = 1e-12))

    Λ_start = rand(4, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, buffers)
    @test all(isapprox.(Λ_end, Λ_start; atol = 1e-12))
end

@testset "AssetPriceChangeStage — custom field names" begin
    layout = StateLayout(
        StateAxis(:b, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:k, [0.0, 1.0, 2.0]),
    )
    stage = AssetPriceChangeStage(layout;
        holdings_axis = :k,
        wealth_axis   = :b,
        q_field       = :p_now,
        q_last_field  = :p_prev,
    )
    @test stage.wealth_axis === :b

    env = (; p_now = 1.5, p_prev = 1.0)
    buffers = allocate(stage)
    V_end = reshape(Float64.(1:12), (4, 3))
    @test backward!(stage, V_end, env, buffers) isa AbstractArray
end
