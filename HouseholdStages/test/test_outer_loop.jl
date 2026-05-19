using Test
using HouseholdStages


# A tiny Aiyagari-shaped chain — 2 income states, 20-point wealth grid —
# small enough that VFI converges in tens of iterations and the
# steady-state inner solves are sub-second.

function _tiny_aiyagari_layout(; N_w::Int = 20)
    return StateLayout(
        StateAxis(:wealth, continuous_grid(0.0, 30.0; length = N_w, spacing = :log)),
        StateAxis(:income, [0.6, 1.4]),
    )
end

function _tiny_aiyagari_household(layout::StateLayout;
                                  β::Float64 = 0.96, σ::Float64 = 1.5)
    P_y = [0.7 0.3; 0.3 0.7]
    income_shock = MarkovStage(layout; axis = :income, transition = P_y)
    income_receipt = WealthChangeStage(layout;
        wealth_post  = (cell; env) -> (1 + env.r) * cell.wealth + env.w * cell.income,
        wealth_axis  = :wealth,
    )
    u(c) = c < 0 ? -Inf : (σ == 1.0 ? log(c) : (c^(1 - σ)) / (1 - σ))
    savings = ConsumptionSavingsStage(layout;
        β               = β,
        utility         = (cell, c; env) -> u(c),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
    chain = income_shock ∘ₛ income_receipt ∘ₛ savings
    return lift_moments(chain;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end

function _tiny_aiyagari_prices(K::Real; α::Float64 = 0.36, δ::Float64 = 0.08, L::Float64 = 1.0, A::Real = 1.0)
    r = α * A * (K / L)^(α - 1) - δ
    w = (1 - α) * A * (K / L)^α
    return (; r, w)
end


@testset "solve_vfi_steady_state_given_env! — converges on a tiny chain" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    buffers = allocate(hh)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    res = solve_vfi_steady_state_given_env!(hh, env, buffers;
                                             tol = 1e-6, maxiter = 1000)
    @test res.converged
    @test res.iters > 1
    @test maximum(abs, backward!(hh, res.V, env, buffers) .- res.V) < 1e-5
end


@testset "solve_lambda_steady_state_given_env! — converges to a probability distribution" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    buffers = allocate(hh)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    # Seed kernels by running backward at this env.
    solve_vfi_steady_state_given_env!(hh, env, buffers;
                                       tol = 1e-6, maxiter = 1000)

    res = solve_lambda_steady_state_given_env!(hh, buffers;
                                                tol = 1e-6, maxiter = 50_000)
    @test res.converged
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-8)
    @test all(res.Λ .>= -1e-12)
end


@testset "solve_steady_state_given_env! — bundles V + Λ at one env" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    buffers = allocate(hh)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    res = solve_steady_state_given_env!(hh, env, buffers;
                                         vfi_tol = 1e-6, lambda_tol = 1e-6)
    @test res.vfi_iters > 1
    @test res.lambda_iters > 1
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-8)
    # Bellman fixed-point check
    @test maximum(abs, backward!(hh, res.V, env, buffers) .- res.V) < 1e-5
    # Moments are deliberately not bundled — caller calls compute_moments.
    moments = compute_moments(hh, env)
    @test moments.K_supplied > 0.0
end


@testset "solve_steady_state_given_env! — V_init / Λ_init kwargs warm-start" begin
    layout = _tiny_aiyagari_layout()
    hh = _tiny_aiyagari_household(layout)
    buffers = allocate(hh)
    env = (; K = 5.0, _tiny_aiyagari_prices(5.0)...)

    res1 = solve_steady_state_given_env!(hh, env, buffers;
                                          vfi_tol = 1e-6, lambda_tol = 1e-6)
    # Re-solve with the previous (V, Λ) as warm-start: should converge in
    # vastly fewer iters because we're already at the fixed point.
    res2 = solve_steady_state_given_env!(hh, env, buffers;
                                          V_init = res1.V, Λ_init = res1.Λ,
                                          vfi_tol = 1e-6, lambda_tol = 1e-6)
    @test res2.vfi_iters < res1.vfi_iters
    @test res2.lambda_iters < res1.lambda_iters
end
