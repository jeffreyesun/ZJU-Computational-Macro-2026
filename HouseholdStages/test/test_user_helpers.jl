using Test
using HouseholdStages

# Small Aiyagari-shaped chain reused across helpers tests.
function _aiyagari_chain(; N_w::Int = 20)
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid(0.0, 30.0; length = N_w, spacing = :log)),
        StateAxis(:income, [0.6, 1.4]),
    )
    P_y = [0.7 0.3; 0.3 0.7]
    shock = MarkovStage(layout; axis = :income, transition = P_y)
    receipt = WealthChangeStage(layout;
        wealth_post = (cell; env) -> (1 + env.r) * cell.wealth + env.w * cell.income,
        wealth_axis = :wealth)
    savings = ConsumptionSavingsStage(layout;
        β = 0.96,
        utility = (cell, c; env) -> c < 0 ? -Inf : log(c),
        wealth_axis = :wealth,
        monotone_search = :divide_conquer)
    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh; K_supplied = at_end(integrand = :wealth, reduce = sum))
end

_aiy_prices(K::Real; α = 0.36, δ = 0.08, L = 1.0, A = 1.0) =
    (; r = α * A * (K / L)^(α - 1) - δ, w = (1 - α) * A * (K / L)^α)


@testset "solve_steady_state_given_env! — returns V/Λ/moments/history/iters" begin
    hh  = _aiyagari_chain()
    env = (; _aiy_prices(5.0)...)
    res = solve_steady_state_given_env!(hh, env)
    @test haskey(res, :V) && haskey(res, :Λ)
    @test haskey(res, :moments) && haskey(res, :history) && haskey(res, :iters)
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-6)
    @test res.moments.K_supplied > 0
    @test res.iters > 0
end


@testset "solve_steady_state_given_env! — V/Λ in return are independent copies" begin
    hh  = _aiyagari_chain()
    env = (; _aiy_prices(5.0)...)
    res = solve_steady_state_given_env!(hh, env)
    V_returned = res.V
    Λ_returned = res.Λ

    # Mutate the chain's buffer; the returned copies should not be affected.
    V_start_buffer(hh) .= 0
    @test V_returned[1, 1] != 0 || all(iszero, V_returned)   # buffer changed, copy didn't
end


@testset "solve_steady_state_given_env! — second call warm-starts (faster)" begin
    hh  = _aiyagari_chain()
    env = (; _aiy_prices(5.0)...)
    res1 = solve_steady_state_given_env!(hh, env)
    res2 = solve_steady_state_given_env!(hh, env)   # warm-start from buffer
    @test res2.history.vfi_iters ≤ res1.history.vfi_iters
end


@testset "solve_transition_given_env_path! — degenerate case (constant env_path) leaves SS unchanged" begin
    hh  = _aiyagari_chain()
    env = (; _aiy_prices(5.0)...)
    ss  = solve_steady_state_given_env!(hh, env)

    # Constant env path of length T; ss should be a fixed point of the dynamics.
    T_steps = 5
    env_path = [env for _ in 1:T_steps]

    tr = solve_transition_given_env_path!(hh, env_path; Λ_0 = ss.Λ, V_T = ss.V)
    @test length(tr.V_path)   == T_steps + 1
    @test length(tr.Λ_path)   == T_steps + 1
    @test length(tr.moments_path) == T_steps

    # At a steady state the moment path should be (approximately) constant.
    K_t = [m.K_supplied for m in tr.moments_path]
    @test all(isapprox.(K_t, K_t[1]; atol = 1e-3))
end


@testset "solve_transition_given_env_path! — kernel re-seat works (per-period buffers)" begin
    # The L05 footgun fix: in a transition path, every period's forward
    # uses its own kernel. With per-period chains sharing the Spec, the
    # forward at period t reads the period-t kernel populated by the
    # period-t backward.

    hh  = _aiyagari_chain()
    env_ss = (; _aiy_prices(5.0)...)
    ss  = solve_steady_state_given_env!(hh, env_ss)

    # Two distinct envs; chain should not error.
    env_a = (; _aiy_prices(5.0)...)
    env_b = (; _aiy_prices(5.5)...)
    env_path = [env_a, env_b, env_a, env_b]
    tr = solve_transition_given_env_path!(hh, env_path; Λ_0 = ss.Λ, V_T = ss.V)
    @test isfinite(tr.moments_path[end].K_supplied)
end


@testset "compute_direct_jacobian! — runs without error on a chain with moments" begin
    hh  = _aiyagari_chain()
    env = (; _aiy_prices(5.0)...)
    solve_steady_state_given_env!(hh, env)

    J = compute_direct_jacobian!(hh, env, 5;
                                 inputs = (:r,), outputs = (:K_supplied,))
    @test J isa AbstractMatrix
    @test size(J) == (5, 5)
end


@testset "make_env / env_schema — closures don't surface in static schema" begin
    # For closure-bound stages (WealthChange / ConsumptionSavings) the
    # static schema is empty: env keys are read by user closures, which
    # the package does not introspect. `make_env` is therefore
    # permissive (no required fields to enforce). Errors for missing
    # closure-bound fields surface at the first backward!/forward! call
    # via getproperty failure. This is documented behavior — see
    # `abstract.jl` `effective_env_slice`. The schema-as-contract is
    # only useful for stages with `Param` swept fields or explicit
    # `static_env_deps`.
    hh = _aiyagari_chain()
    schema = env_schema(hh)
    @test schema isa NamedTuple
    # Aiyagari has no swept Params and no static_env_deps; schema is empty.
    @test isempty(schema)

    env = make_env(hh; r = 0.05, w = 1.2)
    @test env.r == 0.05
    @test env.w == 1.2
end


@testset "define_moments! — append-only by default; overwrite kwarg works" begin
    hh = MarkovStage(
        StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])));
        axis = :z, transition = [0.5 0.5; 0.5 0.5],
    )
    chain = ChainStage((hh,))
    define_moment!(chain, :K, at_end(integrand = :z, reduce = sum))
    @test_throws ErrorException define_moment!(chain, :K, at_end(integrand = :z, reduce = sum))

    # Overwrite via opt-in kwarg.
    spec_new = at_end(integrand = :z, reduce = sum)
    define_moment!(chain, :K, spec_new;
                   overwrite_existing_moment_definitions = true)
    @test chain.spec.moments[:K] === spec_new
end
