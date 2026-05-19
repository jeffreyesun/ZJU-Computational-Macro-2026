using Test
using HouseholdStages

@testset "product — uniform 2-component, axis = :group" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s = MarkovStage(layout; axis = :z, transition = P)
    ps = product(s, s; axis = :group)
    @test ps isa ProductStage
    @test ps.axis === :group
    @test ps.axis_size == 2
    @test layout_size(ps.input_layout) == (2, 2)
    @test layout_size(ps.output_layout) == (2, 2)
end

@testset "product — backward & forward run per-component" begin
    P = [0.8 0.2; 0.2 0.8]
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s = MarkovStage(layout; axis = :z, transition = P)
    ps = product(s, s; axis = :group)
    buffers = allocate(ps)

    V_end = ones(2, 2)
    V_start = backward!(ps, V_end, NamedTuple(), buffers)
    @test all(isapprox.(V_start, 1.0; atol = 1e-12))

    # Per-component Λ_start summing to 1 each → Λ_end sums per component.
    Λ_start = rand(2, 2)
    Λ_start[:, 1] ./= sum(Λ_start[:, 1])
    Λ_start[:, 2] ./= sum(Λ_start[:, 2])
    Λ_end = forward!(ps, Λ_start, buffers)
    @test isapprox(sum(Λ_end[:, 1]), 1.0; atol = 1e-12)
    @test isapprox(sum(Λ_end[:, 2]), 1.0; atol = 1e-12)
end

@testset "×ₛ infix operator" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.5 0.5; 0.5 0.5]
    s1 = MarkovStage(layout; axis = :z, transition = P)
    s2 = MarkovStage(layout; axis = :z, transition = P)
    ps = s1 ×ₛ s2
    @test ps isa ProductStage
    @test ps.axis_size == 2
end

@testset "replicate_age — N copies along :age" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.5 0.5; 0.5 0.5]
    s = MarkovStage(layout; axis = :z, transition = P)
    ages = replicate_age(s, 5)
    @test ages.axis === :age
    @test ages.axis_size == 5
    @test layout_size(ages.input_layout) == (2, 5)
end

@testset "product — view-based fused tensor for uniform MarkovStage" begin
    # With use_views=true (default), the rebuilt MarkovStage components
    # should have V_start/Λ_end as SubArray views into the fused tensor.
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.7 0.3; 0.3 0.7]
    s1 = MarkovStage(layout; axis = :z, transition = P)
    s2 = MarkovStage(layout; axis = :z, transition = P)
    ps = product(s1, s2; axis = :group)
    @test ps.components[1].V_start isa SubArray
    @test ps.components[2].V_start isa SubArray
    # The component's V_start is a slice of ps.V_start; writing to one
    # should mutate the other.
    fill!(ps.components[1].V_start, 7.0)
    @test all(ps.V_start[:, 1] .== 7.0)
end

@testset "product — use_views=false falls back to copy-based" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.7 0.3; 0.3 0.7]
    s1 = MarkovStage(layout; axis = :z, transition = P)
    s2 = MarkovStage(layout; axis = :z, transition = P)
    ps = product(s1, s2; axis = :group, use_views = false)
    # Components keep their original Array-typed buffers.
    @test ps.components[1].V_start isa Array
    @test ps.components[1] === s1  # same instance reused
end

@testset "product — IdentityStage view-mode" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    s1 = IdentityStage(layout)
    s2 = IdentityStage(layout)
    ps = product(s1, s2; axis = :group)
    @test ps.components[1].V_start isa SubArray
    fill!(ps.components[1].V_start, 3.0)
    @test all(ps.V_start[:, 1] .== 3.0)
end

@testset "product — ForgetfulSumStage view-mode (V_start views; Λ_end allocates fresh)" begin
    layout = StateLayout(
        StateAxis(:w, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:t, categorical([:a, :b])),
    )
    s1 = ForgetfulSumStage(layout; forget_axis = :t)
    s2 = ForgetfulSumStage(layout; forget_axis = :t)
    ps = product(s1, s2; axis = :group)
    # V_start is the input-layout shape so the product-axis slice matches
    # → V_start uses a view.
    @test ps.components[1].V_start isa SubArray
    # Λ_end is the output-layout shape (different from input minus the
    # product axis); ForgetfulSumStage allocates Λ_end fresh in this case.
    @test ps.components[1].Λ_end isa Array
end

@testset "product — choice stages now use view-mode too" begin
    # ArgmaxStage / LogitChoiceStage / GridSavings were ported to the view
    # protocol in round 4. Verify ArgmaxStage components have SubArray
    # V_start/Λ_end. The `policy` field remains Array{Int,N} since
    # per-component policies don't share fused storage.
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    fp  = (cell, a; env) -> Float64(a)
    nsx = (cell, a) -> a
    s1 = ArgmaxStage(layout; choice_axis = :a, flow_payoff = fp, next_state_idx = nsx)
    s2 = ArgmaxStage(layout; choice_axis = :a, flow_payoff = fp, next_state_idx = nsx)
    ps = product(s1, s2; axis = :group)
    @test ps.components[1].V_start isa SubArray
    @test ps.components[1].Λ_end   isa SubArray
    @test ps.components[1].policy  isa Array
    # Mutating a component's V_start should reflect in the fused tensor.
    fill!(ps.components[1].V_start, 9.0)
    @test all(ps.V_start[:, 1] .== 9.0)
end

@testset "product — heterogeneous types error" begin
    layout = StateLayout(StateAxis(:z, discrete_finite([0.5, 1.5])))
    P = [0.5 0.5; 0.5 0.5]
    s1 = MarkovStage(layout; axis = :z, transition = P)
    s2 = IdentityStage(layout)
    @test_throws ErrorException product(s1, s2; axis = :group)
end

@testset "product — env slice unions component slices" begin
    # NB: components must share Julia *type*, which means sharing the
    # closure object identities — two different `(cell, a; env) -> …`
    # lambdas at different syntactic positions are different types.
    layout = StateLayout(StateAxis(:a, discrete_finite([1, 2])))
    fp  = (cell, a; env) -> Float64(a)
    nsx = (cell, a) -> a
    s1 = LogitChoiceStage(layout;
        choice_axis = :a, flow_payoff = fp, next_state_idx = nsx,
        ε = Param(:ξ1),
    )
    s2 = LogitChoiceStage(layout;
        choice_axis = :a, flow_payoff = fp, next_state_idx = nsx,
        ε = Param(:ξ2),
    )
    ps = product(s1, s2; axis = :age)
    slice = effective_env_slice(ps)
    @test :ξ1 in slice
    @test :ξ2 in slice
end
