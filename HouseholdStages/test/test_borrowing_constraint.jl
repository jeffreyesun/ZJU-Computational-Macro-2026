using Test
using HouseholdStages

@testset "BorrowingConstraint — array mask sets V to -Inf on infeasible cells" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0])),
        StateAxis(:y,      discrete_finite([0.5, 1.0])),
    )
    mask = falses(4, 2)
    mask[1, :] .= true   # wealth = -1 infeasible
    stage = BorrowingConstraint(layout; infeasible = mask)

    kernel, scratch = allocate(stage)
    V_end = reshape(Float64.(1:8), (4, 2))
    V_start = backward!(stage, V_end, NamedTuple(), kernel, scratch)

    # Infeasible cells: -Inf. Feasible cells: passed through.
    @test V_start[1, 1] == -Inf
    @test V_start[1, 2] == -Inf
    @test V_start[2:end, :] == V_end[2:end, :]
end

@testset "BorrowingConstraint — forward is identity on Λ" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:z,      discrete_finite([1, 2])),
    )
    mask = falses(3, 2); mask[1, :] .= true
    stage = BorrowingConstraint(layout; infeasible = mask)

    kernel, scratch = allocate(stage)
    Λ_start = rand(3, 2); Λ_start ./= sum(Λ_start)
    Λ_end = forward!(stage, Λ_start, kernel, scratch)
    @test Λ_end == Λ_start
    @test Λ_end !== Λ_start         # written into the stage's buffer
    @test sum(Λ_end) ≈ sum(Λ_start)
end

@testset "BorrowingConstraint — closure form materialises mask from env" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0])),
        StateAxis(:y,      discrete_finite([0.5, 1.0])),
    )
    # State + env-dependent constraint: infeasible if wealth < env.w_min.
    # BorrowingConstraint passes env as a Ref (matching the WealthChange
    # convention), so the closure unwraps with env[].
    stage = BorrowingConstraint(layout;
        infeasible   = (cell; env) -> cell.wealth < env[].w_min,
        closure_deps = (:w_min,),
    )
    @test :w_min in effective_env_slice(stage)

    kernel, scratch = allocate(stage)
    @test haskey(kernel, :mask)
    V_end = reshape(Float64.(1:8), (4, 2))

    # With w_min = 0: wealth = -1 is infeasible.
    V_start1 = copy(backward!(stage, V_end, (; w_min = 0.0), kernel, scratch))
    @test V_start1[1, :] == [-Inf, -Inf]
    @test V_start1[2:end, :] == V_end[2:end, :]

    # With w_min = 0.5: wealth ∈ {-1, 0} infeasible; the same struct picks up
    # the new env without reconstruction.
    V_start2 = copy(backward!(stage, V_end, (; w_min = 0.5), kernel, scratch))
    @test V_start2[1, :] == [-Inf, -Inf]
    @test V_start2[2, :] == [-Inf, -Inf]
    @test V_start2[3:end, :] == V_end[3:end, :]
end

@testset "BorrowingConstraint — closure and array forms agree" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0, 3.0])),
        StateAxis(:y,      discrete_finite([0.5, 1.0, 1.5])),
    )
    # Both forms encode "wealth < 0 is infeasible."
    mask = [w < 0.0 for w in [-1.0, 0.0, 1.0, 2.0, 3.0], y in [0.5, 1.0, 1.5]]
    stage_arr = BorrowingConstraint(layout; infeasible = mask)
    stage_fn  = BorrowingConstraint(layout;
        infeasible = (cell; env) -> cell.wealth < 0.0,   # env unused; no need to unwrap
    )

    k_a, sc_a = allocate(stage_arr)
    k_f, sc_f = allocate(stage_fn)
    V_end = randn(5, 3)
    V_arr = copy(backward!(stage_arr, V_end, NamedTuple(), k_a, sc_a))
    V_fn  = copy(backward!(stage_fn,  V_end, NamedTuple(), k_f, sc_f))
    @test V_arr == V_fn
end

@testset "BorrowingConstraint — duality identity (excluding -Inf cells)" begin
    # ⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩ where r is the (degenerate) flow payoff.
    # With Λ_out = Λ_in and r ≡ 0 on feasible cells (V_in = V_end on feasible),
    # the identity reduces to V_in · Λ_in = V_out · Λ_in on feasible cells.
    # We test that: when Λ has no mass on infeasible cells, ⟨V_in, Λ⟩ = ⟨V_end, Λ⟩.
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:y,      discrete_finite([0.5, 1.0])),
    )
    mask = falses(4, 2); mask[1, :] .= true
    stage = BorrowingConstraint(layout; infeasible = mask)

    kernel, scratch = allocate(stage)
    V_end = randn(4, 2)
    Λ = rand(4, 2); Λ[1, :] .= 0.0    # no mass on infeasible cells
    Λ ./= sum(Λ)

    V_start = backward!(stage, V_end, NamedTuple(), kernel, scratch)
    Λ_end   = forward!(stage, Λ, kernel, scratch)
    # Replace V_start's -Inf entries with 0 before the dot-product;
    # they're multiplied by zero mass anyway.
    V_start_safe = ifelse.(isfinite.(V_start), V_start, 0.0)
    @test isapprox(sum(V_start_safe .* Λ), sum(V_end .* Λ_end); atol = 1e-12)
end

@testset "BorrowingConstraint — composition with MarkovAlong" begin
    # Place the constraint AFTER MarkovAlong:  chain = markov ∘ₛ constraint
    # Backward walks reverse: constraint first (sets V_start at -Inf for
    # infeasible cells), then markov mixes those -Inf values into upstream V.
    P = [0.7 0.3; 0.3 0.7]
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([-1.0, 0.0, 1.0, 2.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    markov = MarkovAlong(layout; axis = :z, transition = P)
    mask   = falses(4, 2); mask[1, :] .= true
    bc     = BorrowingConstraint(layout; infeasible = mask)

    chain = markov ∘ₛ bc
    caches, scratches = allocate(chain, Float64)
    V_end = ones(4, 2)
    V_start = backward!(chain, V_end, NamedTuple(), caches, scratches)
    # First wealth row of bc.V_start is -Inf in both income columns;
    # Markov mixes those across z, so the first row of V_start is -Inf
    # in *both* z columns.
    @test all(V_start[1, :] .== -Inf)
    # Feasible rows: V passes through MarkovAlong with Ps[1, :] · [1, 1] = 1.
    @test all(isapprox.(V_start[2:end, :], 1.0; atol = 1e-12))
end

@testset "BorrowingConstraint — static_env_deps" begin
    @test static_env_deps(BorrowingConstraint) === NamedTuple()
end

@testset "BorrowingConstraint — shape check on the array form" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:y,      discrete_finite([0.5, 1.0])),
    )
    @test_throws ErrorException BorrowingConstraint(layout; infeasible = falses(4, 3))
end
