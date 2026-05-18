using Test
using HouseholdStages

# Tests for the two monotone-policy argmax helpers in
# `src/helper/interpolations.jl` and the corresponding `:sequential` vs
# `:divide_conquer` paths through `ConsumptionSavings.backward!`.

@testset "k1_argmax_dc! matches k1_argmax_monotone! on monotone-policy inputs" begin
    # Random-ish but monotone policy: u_slice[a, s] = -|a - target_policy(s)|
    # makes the argmax a known monotone function of s.
    n_a = 17
    n_s = 23
    target = Int.(round.(range(1, n_a; length = n_s)))
    u_slice = [-abs(a - target[s]) + 0.0 for a in 1:n_a, s in 1:n_s]
    V_post  = zeros(Float64, n_a)

    V_prec_seq = zeros(Float64, n_s)
    policy_seq = zeros(Int, n_s)
    k1_argmax_monotone!(V_prec_seq, policy_seq, u_slice, V_post)

    V_prec_dc = zeros(Float64, n_s)
    policy_dc = zeros(Int, n_s)
    k1_argmax_dc!(V_prec_dc, policy_dc, u_slice, V_post)

    @test policy_seq == policy_dc
    @test policy_dc == target
    @test V_prec_seq ≈ V_prec_dc atol = 1e-12
end

@testset "k1_argmax_dc! handles a non-power-of-two grid (reference required ispow2(n-1))" begin
    n_a = 50
    n_s = 50         # n_s - 1 = 49 is not a power of two
    target = Int.(round.(range(1, n_a; length = n_s)))
    u_slice = [-abs(a - target[s]) + 0.0 for a in 1:n_a, s in 1:n_s]
    V_post  = zeros(Float64, n_a)
    V_prec  = zeros(Float64, n_s)
    policy  = zeros(Int, n_s)

    k1_argmax_dc!(V_prec, policy, u_slice, V_post)
    @test policy == target
end

@testset "ConsumptionSavings — :divide_conquer matches :sequential on the Aiyagari calibration" begin
    # Small Aiyagari-shape problem; exponential wealth grid; CRRA log utility.
    n_w = 64
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid(
            [exp(t) - 1.0 for t in range(0.0, log(101.0); length = n_w)])),
        StateAxis(:income, discrete_finite([0.6, 1.0, 1.4])),
    )
    u = (cell, c; env) -> log(c)
    seq = ConsumptionSavings(layout; β = 0.96, utility = u,
                                       wealth_axis = :wealth,
                                       monotone_search = :sequential)
    dc  = ConsumptionSavings(layout; β = 0.96, utility = u,
                                       wealth_axis = :wealth,
                                       monotone_search = :divide_conquer)

    V_end = [0.1 * w_i + 0.05 * y_j for w_i in 1:n_w, y_j in 1:3]
    env   = NamedTuple()

    cache_seq, scratch_seq = allocate(seq)
    cache_dc,  scratch_dc  = allocate(dc)
    V_seq = copy(backward!(seq, V_end, env, cache_seq, scratch_seq))
    V_dc  = copy(backward!(dc,  V_end, env, cache_dc,  scratch_dc))

    @test seq.policy == dc.policy
    @test V_seq ≈ V_dc atol = 1e-12
end

@testset "ConsumptionSavings — :divide_conquer rejects unknown search mode" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:y,      discrete_finite([1.0])),
    )
    @test_throws ErrorException ConsumptionSavings(layout;
        β = 0.96,
        utility = (cell, c; env) -> log(c),
        monotone_search = :something_else,
    )
end
