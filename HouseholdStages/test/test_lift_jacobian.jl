using Test
using HouseholdStages
using ForwardDiff
using ForwardDiff: Dual, Tag

@testset "lift_jacobian — :reverse returns stage; uses per-stage adjoints" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.7 0.3; 0.3 0.7]
    s = MarkovStage(layout; axis = :z, transition = P)
    @test lift_jacobian(s; mode = :reverse) === s
end

@testset "lift_jacobian — unknown mode errors" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.7 0.3; 0.3 0.7]
    s = MarkovStage(layout; axis = :z, transition = P)
    @test_throws ErrorException lift_jacobian(s; mode = :sideways)
end

@testset "with_eltype — buffer eltype changes; static fields shared" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.7 0.3; 0.3 0.7]
    s = MarkovStage(layout; axis = :z, transition = P)
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    s_d = with_eltype(s, D)
    @test eltype(s_d.buffer.V_start) === D
    @test eltype(s_d.buffer.Λ_end)   === D
    @test s_d.spec.transition       === P            # static field shared
    @test s_d.spec.input_layout     === s.spec.input_layout
    @test s_d.spec.axis             === s.spec.axis
end

@testset "lift_jacobian(MarkovStage) — Dual flows through backward correctly" begin
    # The simplest possible test: MarkovStage is V_θ-independent, so the
    # Jacobian of V_in wrt V_end is exactly P (the transition matrix
    # along the dim, transposed in the Markov-rows-are-conditioning
    # convention). Verify a single tangent direction.
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.7 0.3; 0.3 0.7]
    s = MarkovStage(layout; axis = :z, transition = P)
    s_d = lift_jacobian(s; mode = :forward, n_dual = 1)

    # V_end has a tangent vector ξ along the z axis: ξ = e_1 = (1, 0).
    # V_end[i] = 0 + ξ[i]·ε  (primal 0, partial 1 if i=1 else 0)
    D = eltype(s_d.buffer.V_start)
    V_end = D[
        D(0.0, ForwardDiff.Partials((1.0,))),
        D(0.0, ForwardDiff.Partials((0.0,))),
    ]
    V_start = backward!(s_d, V_end, NamedTuple())
    # V_start = P^T V_end → V_start_dual.partials should equal P^T * e_1
    # = [P[1,1], P[1,2]] = [0.7, 0.3] (rows of P sum to 1 == row-major).
    # Actually for our convention rows=conditioning, V_start[i] = Σ_j P[i,j]*V_end[j],
    # so dV_start/dV_end[1] = column 1 of P = [P[1,1], P[2,1]] = [0.7, 0.3].
    @test ForwardDiff.partials(V_start[1])[1] ≈ P[1, 1] atol = 1e-12
    @test ForwardDiff.partials(V_start[2])[1] ≈ P[2, 1] atol = 1e-12
    @test ForwardDiff.value(V_start[1]) ≈ 0.0 atol = 1e-12
    @test ForwardDiff.value(V_start[2]) ≈ 0.0 atol = 1e-12
end

@testset "lift_jacobian(3-stage chain) — ∂K_supplied/∂r matches finite diffs" begin
    # 4-wealth × 2-income layout, exercising forward-mode AD through the
    # canonical L03/L04 decomposition `IncomeShock ∘ IncomeReceipt ∘
    # ConsumptionSavingsStage`. Hold V_terminal, Λ_init fixed; compute
    # K_supplied(r, w) = Σ Λ_end · wealth. Check ∂K_supplied/∂r via AD
    # against a centered finite difference. (Replaces the prior
    # GridSavings test after the legacy stage was removed.)

    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:income, discrete_finite([0.5, 1.5])),
    )
    P = [0.7 0.3; 0.3 0.7]
    shock   = MarkovStage(layout; axis = :income, transition = P)
    receipt = WealthChangeStage(layout;
        wealth_post  = (cell; env) -> (1 + env.r) * cell.wealth + env.w * cell.income,
        wealth_axis  = :wealth,
    )
    saves = ConsumptionSavingsStage(layout;
        β            = 0.96,
        utility      = (cell, c; env) -> log(c),
        wealth_axis  = :wealth,
    )
    chain = shock ∘ receipt ∘ saves

    function K_supplied(chain_use, r::T, w::T) where {T<:Real}
        # Smooth, non-degenerate V_terminal to avoid argmax kinks near
        # the tangent direction.
        dims = layout_size(layout)
        V_term = T.([0.1 * w_i + 0.05 * y_j for w_i in 1:dims[1], y_j in 1:dims[2]])
        Λ_init = ones(T, dims...) ./ prod(dims)
        backward!(chain_use, V_term, (r = r, w = w))
        Λ_end = forward!(chain_use, Λ_init)
        wealth_grid = T.([0.0, 1.0, 2.0, 3.0])
        s = zero(T)
        for w_i in 1:dims[1], y_j in 1:dims[2]
            s += Λ_end[w_i, y_j] * wealth_grid[w_i]
        end
        return s
    end

    r0, w0 = 0.04, 1.2
    # AD path.
    chain_dual = lift_jacobian(chain; mode = :forward, n_dual = 1)
    D = eltype(chain_dual.buffer.stages[1].V_start)
    r_dual = D(r0, ForwardDiff.Partials((1.0,)))
    w_dual = D(w0, ForwardDiff.Partials((0.0,)))
    K_dual = K_supplied(chain_dual, r_dual, w_dual)
    dKdr_ad = ForwardDiff.partials(K_dual)[1]
    K_primal_ad = ForwardDiff.value(K_dual)

    # Finite-difference path on the Float64 chain.
    h = 1e-6
    K_plus  = K_supplied(chain, r0 + h, w0)
    K_minus = K_supplied(chain, r0 - h, w0)
    K_primal_fd = K_supplied(chain, r0, w0)
    dKdr_fd = (K_plus - K_minus) / (2h)

    @test K_primal_ad ≈ K_primal_fd atol = 1e-10
    @test dKdr_ad ≈ dKdr_fd rtol = 1e-4
end

@testset "backward_adjoint!(MarkovStage) — dot-product test" begin
    # Verify the operator adjointness identity:
    #     ⟨backward!(V_end), dV_start⟩ = ⟨V_end, backward_adjoint!(dV_start)⟩
    # (without the flow payoff r, since MarkovStage has r = 0). This is
    # the cleanest way to confirm the adjoint matches the primal.
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.6 0.4; 0.25 0.75]
    s = MarkovStage(layout; axis = :z, transition = P)

    V_end    = randn(2)
    dV_start = randn(2)
    V_start  = copy(backward!(s, V_end, NamedTuple()))
    dV_end   = backward_adjoint!(s, dV_start)

    @test sum(V_start .* dV_start) ≈ sum(V_end .* dV_end) atol = 1e-12
end

@testset "forward_adjoint!(MarkovStage) — dot-product test" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.6 0.4; 0.25 0.75]
    s = MarkovStage(layout; axis = :z, transition = P)

    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)
    dΛ_end  = randn(2)
    Λ_end   = copy(forward!(s, Λ_start))
    dΛ_start = forward_adjoint!(s, dΛ_end)

    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12
end

@testset "adjoints(IdentityStage) — pass-through" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s = IdentityStage(layout)

    dV = randn(2)
    dΛ = randn(2)
    @test backward_adjoint!(s, dV) == dV
    @test forward_adjoint!(s, dΛ) == dΛ
end

@testset "adjoints(ForgetfulSumStage) — dot-product tests" begin
    layout = StateLayout(
        StateAxis(:w, continuous_grid([1.0, 2.0, 3.0])),
        StateAxis(:t, categorical([:a, :b])),
    )
    s = ForgetfulSumStage(layout; forget_axis = :t)

    # Forward: Λ_start (3,2) → Λ_end (3,).
    Λ_start = abs.(randn(3, 2)); Λ_start ./= sum(Λ_start)
    dΛ_end  = randn(3)
    Λ_end   = copy(forward!(s, Λ_start))
    dΛ_start = forward_adjoint!(s, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    # Backward: V_end (3,) → V_start (3,2).
    V_end    = randn(3)
    dV_start = randn(3, 2)
    V_start  = copy(backward!(s, V_end, NamedTuple()))
    dV_end   = backward_adjoint!(s, dV_start)
    @test sum(V_start .* dV_start) ≈ sum(V_end .* dV_end) atol = 1e-12
end

@testset "adjoints(ArgmaxStage) — dot-product test" begin
    layout = StateLayout(StateAxis(:s, categorical([:A, :B])))
    stage = ArgmaxStage(layout;
        choice_axis    = :s,
        flow_payoff    = (cell, a; env) -> (a == :B ? 1.0 : 0.0),
        next_state_idx = (cell, a) -> a == :A ? 1 : 2,
    )
    V_end = randn(2)
    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)

    V_start = copy(backward!(stage, V_end, NamedTuple()))
    Λ_end   = copy(forward!(stage, Λ_start))

    # Sensitivities at the input and output of each pass.
    dV_in    = randn(2)
    dΛ_end   = randn(2)
    dV_out   = backward_adjoint!(stage, dV_in)
    dΛ_start = forward_adjoint!(stage, dΛ_end)

    # ArgmaxStage flow_payoff at the chosen action is non-zero. The duality
    # check for the BACKWARD pass needs to account for that; we check
    # the *linear* part of the adjoint via the FORWARD pass identity
    # ⟨Λ_end, dΛ_end⟩ = ⟨Λ_start, dΛ_start⟩ (no payoff term).
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    # For backward, ⟨V_in - r, dV_in⟩ = ⟨V_end, dV_out⟩ where r is the
    # flow-payoff contribution. Compute V_in - r explicitly:
    # at every cell, V_in = max(r + V_out[ν]); the chosen action's
    # contribution is r(action*) + V_out[ν(s,action*)]. Subtracting r:
    actions = [:A, :B]
    V_in_minus_r = similar(V_start)
    for (idx, cell) in cells(layout)
        ci  = CartesianIndex(Tuple(idx))
        a_i = stage.buffer.kernel.policy[ci]
        # V_start[ci] = r(action*) + V_end[ν(s,action*)]
        r_val = (actions[a_i] == :B ? 1.0 : 0.0)
        V_in_minus_r[ci] = V_start[ci] - r_val
    end
    @test sum(V_in_minus_r .* dV_in) ≈ sum(V_end .* dV_out) atol = 1e-12
end

@testset "adjoints(LogitChoiceStage) — dot-product test on forward" begin
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    stage = LogitChoiceStage(layout;
        choice_axis    = :a,
        flow_payoff    = (cell, a; env) -> Float64(a),
        next_state_idx = (cell, a) -> a,
        ε              = Param(0.5),
    )
    V_end = randn(2)
    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)

    V_start = copy(backward!(stage, V_end, NamedTuple()))
    Λ_end   = copy(forward!(stage, Λ_start))

    dΛ_end = randn(2)
    dΛ_start = forward_adjoint!(stage, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    # Verify backward_adjoint! against the analytic envelope-theorem
    # derivative: ∂V_in[s]/∂V_end[c'] = Σ_a P(a|s) · I(ν(s,a) = c').
    # For our setup, ν(s,a) = a (the action sets next state), so:
    # ∂V_in[s]/∂V_end[c'] = P(c'|s) = stage.buffer.kernel.choice_prob[s, c'].
    e1 = Float64[1.0, 0.0]
    dV_out_via_adj = backward_adjoint!(stage, e1)
    # backward_adjoint! pushes dV_in's contribution at s=1 backward to
    # all destination cells weighted by P(a|s=1).
    # For dV_in = [1, 0]: dV_out[c'] = P(c'|s=1) (only s=1 contributes).
    @test dV_out_via_adj[1] ≈ stage.buffer.kernel.choice_prob[1, 1] atol = 1e-12
    @test dV_out_via_adj[2] ≈ stage.buffer.kernel.choice_prob[1, 2] atol = 1e-12
end

@testset "adjoints(ConsumptionSavingsStage) — dot-product test on forward" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0])),
        StateAxis(:y, discrete_finite([1.0])),
    )
    stage = ConsumptionSavingsStage(layout;
        β            = 0.95,
        utility      = (cell, c; env) -> log(c),
        wealth_axis  = :wealth,
    )
    backward!(stage, zeros(2, 1), NamedTuple())  # populate policy

    Λ_start = reshape([0.4, 0.6], (2, 1))
    Λ_end = copy(forward!(stage, Λ_start))

    dΛ_end = reshape([1.0, 0.5], (2, 1))
    dΛ_start = forward_adjoint!(stage, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12
end

@testset "adjoints(ChainStage of linear-K stages) — dot-product test" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P1 = [0.7 0.3; 0.4 0.6]
    P2 = [0.5 0.5; 0.2 0.8]
    s1 = MarkovStage(layout; axis = :z, transition = P1)
    s2 = MarkovStage(layout; axis = :z, transition = P2)
    chain = s1 ∘ s2

    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)
    dΛ_end  = randn(2)
    Λ_end   = copy(forward!(chain, Λ_start))
    dΛ_start = forward_adjoint!(chain, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    V_end    = randn(2)
    dV_start = randn(2)
    V_start  = copy(backward!(chain, V_end, NamedTuple()))
    dV_end   = backward_adjoint!(chain, dV_start)
    @test sum(V_start .* dV_start) ≈ sum(V_end .* dV_end) atol = 1e-12
end

@testset "lift_jacobian(ChainStage with moments) — works through define_moments!" begin
    # Verify the lift propagates through `define_moments!` (which now returns
    # a `ChainStage` whose `moments` field is non-empty).
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([1.0, 2.0, 3.0])),
        StateAxis(:income, [0.5, 1.5]),
    )
    P = [0.5 0.5; 0.5 0.5]
    shock = MarkovStage(layout; axis = :income, transition = P)
    mc = define_moments!(shock; K = at_end(integrand = :wealth, reduce = sum))
    mc_d = lift_jacobian(mc; mode = :forward, n_dual = 1)

    @test mc_d isa ChainStage
    @test !isempty(mc_d.spec.moments)                          # moments preserved through with_eltype
    inner_stage = mc_d.spec.stages[1]
    @test eltype(inner_stage.transition) === Float64           # transition stays Float64
    inner_buffer = mc_d.buffer.stages[1]
    @test eltype(inner_buffer.V_start)    !== Float64          # buffer is Dual
end
