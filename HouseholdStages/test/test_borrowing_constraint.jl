using Test
using HouseholdStages

@testset "BorrowingConstraintStage — array mask sets V to -Inf on infeasible cells" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0])),
        StateAxis(:y,      [0.5, 1.0]),
    )
    mask = falses(4, 2)
    mask[1, :] .= true   # wealth = -1 infeasible
    stage = BorrowingConstraintStage(layout; infeasible = mask)

    buffers = allocate(stage)
    V_end = reshape(Float64.(1:8), (4, 2))
    V_start = backward!(stage, V_end, NamedTuple(), buffers)

    @test V_start[1, 1] == -Inf
    @test V_start[1, 2] == -Inf
    @test V_start[2:end, :] == V_end[2:end, :]
end

@testset "BorrowingConstraintStage — forward is identity on Λ" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z,      [1, 2]),
    )
    mask = falses(3, 2); mask[1, :] .= true
    stage = BorrowingConstraintStage(layout; infeasible = mask)

    buffers = allocate(stage)
    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, buffers)
    @test Λ_end == Λ_start
    @test Λ_end !== Λ_start
    @test sum(Λ_end) ≈ sum(Λ_start)
end

@testset "BorrowingConstraintStage — closure form materialises mask from env" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0])),
        StateAxis(:y,      [0.5, 1.0]),
    )
    # State + env-dependent constraint: infeasible if wealth < env.w_min.
    # env is passed directly to the closure (no Ref wrapping).
    stage = BorrowingConstraintStage(layout;
        infeasible = (cell; env) -> cell.wealth < env.w_min,
    )

    buffers = allocate(stage)
    @test haskey(buffers.kernel, :mask)
    V_end = reshape(Float64.(1:8), (4, 2))

    # With w_min = 0: wealth = -1 is infeasible.
    V_start1 = copy(backward!(stage, V_end, (; w_min = 0.0), buffers))
    @test V_start1[1, :] == [-Inf, -Inf]
    @test V_start1[2:end, :] == V_end[2:end, :]

    # With w_min = 0.5: wealth ∈ {-1, 0} infeasible.
    V_start2 = copy(backward!(stage, V_end, (; w_min = 0.5), buffers))
    @test V_start2[1, :] == [-Inf, -Inf]
    @test V_start2[2, :] == [-Inf, -Inf]
    @test V_start2[3:end, :] == V_end[3:end, :]
end

@testset "BorrowingConstraintStage — closure and array forms agree" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0, 3.0])),
        StateAxis(:y,      [0.5, 1.0, 1.5]),
    )
    mask = [w < 0.0 for w in [-1.0, 0.0, 1.0, 2.0, 3.0], y in [0.5, 1.0, 1.5]]
    stage_arr = BorrowingConstraintStage(layout; infeasible = mask)
    stage_fn  = BorrowingConstraintStage(layout;
        infeasible = (cell; env) -> cell.wealth < 0.0,
    )

    buf_a = allocate(stage_arr)
    buf_f = allocate(stage_fn)
    V_end = randn(5, 3)
    V_arr = copy(backward!(stage_arr, V_end, NamedTuple(), buf_a))
    V_fn  = copy(backward!(stage_fn,  V_end, NamedTuple(), buf_f))
    @test V_arr == V_fn
end

@testset "BorrowingConstraintStage — duality identity (excluding -Inf cells)" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:y,      [0.5, 1.0]),
    )
    mask = falses(4, 2); mask[1, :] .= true
    stage = BorrowingConstraintStage(layout; infeasible = mask)

    buffers = allocate(stage)
    V_end = randn(4, 2)
    Λ = rand(4, 2); Λ[1, :] .= 0.0
    Λ ./= sum(Λ)

    V_start = backward!(stage, V_end, NamedTuple(), buffers)
    Λ_end   = forward!(stage, Λ, buffers)
    V_start_safe = ifelse.(isfinite.(V_start), V_start, 0.0)
    @test isapprox(sum(V_start_safe .* Λ), sum(V_end .* Λ_end); atol = 1e-12)
end

@testset "BorrowingConstraintStage — composition with MarkovStage" begin
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0])),
        StateAxis(:z,      [0.5, 1.5]),
    )
    markov = MarkovStage(layout; axis = :z, transition = P)
    mask   = falses(4, 2); mask[1, :] .= true
    bc     = BorrowingConstraintStage(layout; infeasible = mask)

    chain = markov ∘ₛ bc
    buffers = allocate(chain, Float64)
    V_end = ones(4, 2)
    V_start = backward!(chain, V_end, NamedTuple(), buffers)
    @test all(V_start[1, :] .== -Inf)
    @test all(isapprox.(V_start[2:end, :], 1.0; atol = 1e-12))
end

@testset "BorrowingConstraintStage — static_env_deps" begin
    @test static_env_deps(BorrowingConstraintStage) === NamedTuple()
end

@testset "BorrowingConstraintStage — shape check on the array form" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:y,      [0.5, 1.0]),
    )
    @test_throws ErrorException BorrowingConstraintStage(layout; infeasible = falses(4, 3))
end
