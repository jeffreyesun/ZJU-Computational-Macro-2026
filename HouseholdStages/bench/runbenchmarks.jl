###################################################################
# HouseholdStages benchmarks vs. hand-coded reference implementations
###################################################################
#
# Goal: measure each library stage's backward!/forward! against a
# hand-coded reference written in the style of
# `reference_materials/example_stages/` (fixed shapes, direct buffer
# mutation, no closure dispatch, no per-call allocation). Ratios well
# below ~4× are acceptable; ratios above ~10× call for investigation.
#
# Run:
#     julia --project=HouseholdStages/bench HouseholdStages/bench/runbenchmarks.jl
#
# The script prints a table to stdout and writes the same table to
# `bench/results.txt`. Pass `LIBRARY_ONLY` as an env var to skip the
# reference baselines (useful when iterating on library fixes).
# Pass `LARGE_BENCH=1` to re-run at the example-driver sizes
# (N_w = 400 everywhere) — slower, more representative; written into
# `bench/results.md` alongside the default-size run.

using BenchmarkTools, Printf, LinearAlgebra, Random
using HouseholdStages

const NW_BASE = 80
const NZ_BASE = 3
const NW_KS   = 100
const NZ_KS   = 2
const NW_SPATIAL = 60
const NZ_SPATIAL = 3
const NL_SPATIAL = 2
const SPATIAL_EPSILON = 5.0
const SPATIAL_MIG_COST = 0.5

# LARGE_BENCH=1 reruns the suite at the example-driver sizes (N_w = 400
# everywhere; spatial layout = (400, 3, 2)). Default off — the small
# sizes give fast iteration feedback. The large numbers are what the
# README and bench/results.md quote for the user-facing claim.
const NW_BASE_LARGE = 400
const NW_KS_LARGE   = 400
const NW_SPATIAL_LARGE = 400

# Fixed parameters shared with the Aiyagari example calibration.
const BETA = 0.96
const SIGMA = 1.5
const R = 0.04
const W = 1.0

const Y_GRID_BASE = [0.6, 1.0, 1.4]
const P_Y_BASE    = [0.7 0.2 0.1;
                     0.2 0.6 0.2;
                     0.1 0.2 0.7]

const Y_GRID_KS = [0.07, 1.0]
const P_Y_KS    = [0.6 0.4;
                   0.04 0.96]


# Exponential wealth grid (matches aiyagari/model.jl::exp_wealth_grid).
function exp_wgrid(lo, hi, n; shift = 1.0)
    return [exp(t) * shift - shift + lo
            for t in range(0.0, log((hi - lo + shift) / shift); length = n)]
end


# Utility shared with the library (CRRA, σ = SIGMA).
@inline u_crra_bench(c) = c > 0 ? c^(1 - SIGMA) / (1 - SIGMA) : -Inf


###############################################################################
# Stage builders (library)
###############################################################################

function build_aiyagari_chain(n_w::Int, n_z::Int, y_grid::Vector{Float64},
                              P_y::Matrix{Float64})
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid(exp_wgrid(0.0, 100.0, n_w))),
        StateAxis(:income, discrete_finite(y_grid)),
    )
    shock = MarkovStage(layout; axis = :income, transition = P_y)
    function wp(cell; env)
        e = env[]
        return (1 + e.r) * cell.wealth + e.w * cell.income
    end
    receipt = WealthChangeStage(layout; wealth_post = wp,
                                   wealth_axis = :wealth,
                                   closure_deps = (:r, :w))
    savings = ConsumptionSavingsStage(layout;
        β = BETA,
        utility = (cell, c; env) -> u_crra_bench(c),
        wealth_axis = :wealth,
    )
    return layout, shock, receipt, savings
end

function build_spatial_chain(n_w::Int, n_z::Int, n_l::Int,
                             y_grid::Vector{Float64},
                             P_y::Matrix{Float64})
    layout = StateLayout(
        StateAxis(:wealth,   continuous_grid(exp_wgrid(0.0, 30.0, n_w))),
        StateAxis(:income,   discrete_finite(y_grid)),
        StateAxis(:location, categorical([Symbol("loc$i") for i in 1:n_l])),
    )
    shock = MarkovStage(layout; axis = :income, transition = P_y)
    C = SPATIAL_MIG_COST .* (ones(n_l, n_l) .- LinearAlgebra.I(n_l))
    move = MigrationStage(layout;
        location_axis  = :location,
        migration_cost = C,
        ε              = SPATIAL_EPSILON,
    )
    function wp(cell; env)
        e = env[]
        r = cell.location == :loc1 ? e.r_home : e.r_abroad
        w = cell.location == :loc1 ? e.w_home : e.w_abroad
        return (1 + r) * cell.wealth + w * cell.income
    end
    receipt = WealthChangeStage(layout;
        wealth_post  = wp,
        wealth_axis  = :wealth,
        closure_deps = (:r_home, :r_abroad, :w_home, :w_abroad),
    )
    savings = ConsumptionSavingsStage(layout;
        β            = BETA,
        utility      = (cell, c; env) -> u_crra_bench(c),
        wealth_axis  = :wealth,
    )
    return layout, shock, move, receipt, savings
end


###############################################################################
# Hand-coded reference kernels
###############################################################################

# Markov along axis 2 of a (N_w, N_z) array. Reference style: permute z
# to front, mul!, permute back. (Matches the reference's get_V_preshock
# pattern.)
function ref_markov_backward!(V_pre, V_end, P_y,
                              V_end_perm, V_pre_perm)
    permutedims!(V_end_perm, V_end, (2, 1))
    mul!(V_pre_perm, P_y, V_end_perm)        # V_pre_perm[zi, ki] = ∑_zi' P[zi,zi'] V_end_perm[zi',ki]
    permutedims!(V_pre, V_pre_perm, (2, 1))
    return V_pre
end

function ref_markov_forward!(Λ_post, Λ_pre, P_y_T,
                             Λ_pre_perm, Λ_post_perm)
    permutedims!(Λ_pre_perm, Λ_pre, (2, 1))
    mul!(Λ_post_perm, P_y_T, Λ_pre_perm)
    permutedims!(Λ_post, Λ_post_perm, (2, 1))
    return Λ_post
end

# WealthChangeStage backward — hand-coded per-column linear interp.
function ref_wealthchange_backward!(V_pre, V_end, wgrid, wpost,
                                    ::Val{extrap}) where {extrap}
    # V_pre[wi, zi] = lookup V_end[:, zi] at wpost[wi, zi].
    for zi in axes(V_end, 2)
        v_end_col = @view V_end[:, zi]
        wpost_col = @view wpost[:, zi]
        v_pre_col = @view V_pre[:, zi]
        reinterpolate!(v_pre_col, v_end_col, wgrid, wpost_col, Val(extrap))
    end
    return V_pre
end

function ref_wealthchange_forward!(Λ_post, Λ_pre, wgrid, wpost)
    for zi in axes(Λ_pre, 2)
        lp_col   = @view Λ_pre[:, zi]
        lpost_col = @view Λ_post[:, zi]
        wpost_col = @view wpost[:, zi]
        convert_distribution!(lpost_col, lp_col, wpost_col, wgrid, Val(:share))
    end
    return Λ_post
end

# ConsumptionSavingsStage backward — hand-coded monotone-policy walk.
function ref_cs_backward!(V_pre, policy, V_end, wgrid, β)
    n_w, n_z = size(V_end)
    for zi in 1:n_z
        prev_a = 1
        for w_in_i in 1:n_w
            b_in = wgrid[w_in_i]
            best_v = -Inf
            best_a = 0
            for a_i in prev_a:n_w
                c = b_in - wgrid[a_i]
                c > 0 || continue
                u = u_crra_bench(c)
                v = u + β * V_end[a_i, zi]
                if v > best_v
                    best_v = v
                    best_a = a_i
                end
            end
            if best_a == 0
                V_pre[w_in_i, zi] = -Inf
                policy[w_in_i, zi] = 1
            else
                V_pre[w_in_i, zi] = best_v
                policy[w_in_i, zi] = best_a
                prev_a = best_a
            end
        end
    end
    return V_pre
end

# MigrationStage backward — hand-coded logit over the location axis for the
# spatial chain. Three-pass log-sum-exp (max for stability, weights,
# normalise). Reference style: tight inner loop, fixed shapes.
function ref_migration_backward!(V_pre, prob, V_end, C, ε)
    n_w, n_z, n_l = size(V_end)
    for zi in 1:n_z, w_i in 1:n_w
        # Pass 1: max across destinations.
        max_u = -Inf
        for i in 1:n_l, j in 1:n_l
            u = -C[i, j] + V_end[w_i, zi, j]
            u > max_u && (max_u = u)
        end
        # Pass 2: per-origin weights + V_pre = max + ε log Z.
        for i in 1:n_l
            denom = 0.0
            for j in 1:n_l
                u = -C[i, j] + V_end[w_i, zi, j]
                w = exp((u - max_u) / ε)
                prob[w_i, zi, i, j] = w
                denom += w
            end
            for j in 1:n_l
                prob[w_i, zi, i, j] /= denom
            end
            V_pre[w_i, zi, i] = max_u + ε * log(denom)
        end
    end
    return V_pre
end

# ConsumptionSavingsStage backward — 3D version for spatial layout.
function ref_cs_backward_3d!(V_pre, policy, V_end, wgrid, β)
    n_w, n_z, n_l = size(V_end)
    for li in 1:n_l, zi in 1:n_z
        prev_a = 1
        for w_in_i in 1:n_w
            b_in = wgrid[w_in_i]
            best_v = -Inf
            best_a = 0
            for a_i in prev_a:n_w
                c = b_in - wgrid[a_i]
                c > 0 || continue
                u = u_crra_bench(c)
                v = u + β * V_end[a_i, zi, li]
                if v > best_v
                    best_v = v
                    best_a = a_i
                end
            end
            if best_a == 0
                V_pre[w_in_i, zi, li] = -Inf
                policy[w_in_i, zi, li] = 1
            else
                V_pre[w_in_i, zi, li] = best_v
                policy[w_in_i, zi, li] = best_a
                prev_a = best_a
            end
        end
    end
    return V_pre
end


###############################################################################
# Benchmark setup helpers
###############################################################################

# Configure a chain + per-stage buffers tuple + warmed V_end / Λ_pre.
# `buffers` is the chain workspace returned by `allocate(chain)` — a
# Tuple of per-stage `(; kernel, scratch)` NamedTuples.
mutable struct BenchSetup
    layout
    shock
    receipt
    savings
    buffers
    V_end :: Matrix{Float64}
    Λ_pre :: Matrix{Float64}
    wpost :: Matrix{Float64}
    wgrid :: Vector{Float64}
    P_y   :: Matrix{Float64}
    env
end

# Spatial variant — 3D layout `(wealth, income, location)`, MigrationStage
# in the chain. Holds everything needed for per-stage and chain
# benchmarks.
mutable struct SpatialBenchSetup
    layout
    shock
    move
    receipt
    savings
    buffers
    V_end :: Array{Float64, 3}
    Λ_pre :: Array{Float64, 3}
    wpost :: Array{Float64, 3}
    wgrid :: Vector{Float64}
    P_y   :: Matrix{Float64}
    C     :: Matrix{Float64}
    env
end

function make_setup(n_w, n_z, y_grid, P_y)
    layout, shock, receipt, savings = build_aiyagari_chain(n_w, n_z, y_grid, P_y)
    chain   = shock ∘ₛ receipt ∘ₛ savings
    buffers = allocate(chain)

    wgrid = exp_wgrid(0.0, 100.0, n_w)
    V_end = [(b + 1) ^ (1 - SIGMA) / (1 - SIGMA) for b in wgrid, _ in 1:n_z]
    Λ_pre = fill(1.0 / (n_w * n_z), n_w, n_z)
    env   = (; K = 5.0, r = R, w = W)

    # Warm kernels by running one backward of the chain at env so kernels
    # carry sensible policies / wealth_post values for the per-stage
    # benchmarks.
    backward!(chain, V_end, env, buffers)

    # Pull the receipt kernel's wealth_post array (post-`(1+r)b + wy`).
    wpost = buffers[2].kernel.wealth_post

    return BenchSetup(layout, shock, receipt, savings,
                      buffers,
                      V_end, Λ_pre, wpost, wgrid, P_y, env)
end

function make_spatial_setup(n_w, n_z, n_l, y_grid, P_y)
    layout, shock, move, receipt, savings =
        build_spatial_chain(n_w, n_z, n_l, y_grid, P_y)
    chain   = shock ∘ₛ move ∘ₛ receipt ∘ₛ savings
    buffers = allocate(chain)

    wgrid = exp_wgrid(0.0, 30.0, n_w)
    V_end = [(b + 1) ^ (1 - SIGMA) / (1 - SIGMA)
             for b in wgrid, _ in 1:n_z, _ in 1:n_l]
    Λ_pre = fill(1.0 / (n_w * n_z * n_l), n_w, n_z, n_l)
    # Two-location calibration; first location plays the "home" role.
    env = (; r_home = 0.04, w_home = 1.0, r_abroad = 0.04, w_abroad = 1.0)

    backward!(chain, V_end, env, buffers)

    wpost = buffers[3].kernel.wealth_post
    C = SPATIAL_MIG_COST .* (ones(n_l, n_l) .- LinearAlgebra.I(n_l))

    return SpatialBenchSetup(layout, shock, move, receipt, savings,
                             buffers,
                             V_end, Λ_pre, wpost, wgrid, P_y, C, env)
end


###############################################################################
# Benchmark cases
###############################################################################

function bench_markov_backward(s::BenchSetup)
    shock = s.shock
    V_end = s.V_end
    env   = s.env
    buf   = s.buffers[1]
    return @benchmark backward!($shock, $V_end, $env, $buf)
end

function bench_markov_backward_ref(s::BenchSetup)
    V_end = s.V_end
    P_y   = s.P_y
    V_pre = similar(V_end)
    V_end_perm = Matrix{Float64}(undef, size(V_end, 2), size(V_end, 1))
    V_pre_perm = Matrix{Float64}(undef, size(V_end, 2), size(V_end, 1))
    return @benchmark ref_markov_backward!($V_pre, $V_end, $P_y, $V_end_perm, $V_pre_perm)
end

function bench_markov_forward(s::BenchSetup)
    shock = s.shock
    Λ_pre = s.Λ_pre
    buf   = s.buffers[1]
    return @benchmark forward!($shock, $Λ_pre, $buf)
end

function bench_markov_forward_ref(s::BenchSetup)
    Λ_pre = s.Λ_pre
    P_y_T = Matrix(s.P_y')
    Λ_post = similar(Λ_pre)
    Λ_pre_perm  = Matrix{Float64}(undef, size(Λ_pre, 2), size(Λ_pre, 1))
    Λ_post_perm = Matrix{Float64}(undef, size(Λ_pre, 2), size(Λ_pre, 1))
    return @benchmark ref_markov_forward!($Λ_post, $Λ_pre, $P_y_T,
                                          $Λ_pre_perm, $Λ_post_perm)
end

function bench_wealthchange_backward(s::BenchSetup)
    receipt = s.receipt
    # The receipt stage's V_end is the output of the savings backward,
    # which is what backward returns from savings. For a per-stage test,
    # any well-typed V_end of the right shape works.
    V_end = s.V_end
    env   = s.env
    buf   = s.buffers[2]
    return @benchmark backward!($receipt, $V_end, $env, $buf)
end

function bench_wealthchange_backward_ref(s::BenchSetup)
    V_end = s.V_end
    wgrid = s.wgrid
    wpost = s.wpost
    V_pre = similar(V_end)
    return @benchmark ref_wealthchange_backward!($V_pre, $V_end, $wgrid, $wpost, Val(:linear))
end

function bench_wealthchange_forward(s::BenchSetup)
    receipt = s.receipt
    Λ_pre = s.Λ_pre
    buf   = s.buffers[2]
    return @benchmark forward!($receipt, $Λ_pre, $buf)
end

function bench_wealthchange_forward_ref(s::BenchSetup)
    Λ_pre = s.Λ_pre
    wgrid = s.wgrid
    wpost = s.wpost
    Λ_post = similar(Λ_pre)
    return @benchmark ref_wealthchange_forward!($Λ_post, $Λ_pre, $wgrid, $wpost)
end

function bench_cs_backward(s::BenchSetup)
    savings = s.savings
    V_end = s.V_end
    env   = s.env
    buf   = s.buffers[3]
    return @benchmark backward!($savings, $V_end, $env, $buf)
end

function bench_cs_backward_ref(s::BenchSetup)
    V_end = s.V_end
    wgrid = s.wgrid
    V_pre = similar(V_end)
    policy = zeros(Int, size(V_end))
    β = BETA
    return @benchmark ref_cs_backward!($V_pre, $policy, $V_end, $wgrid, $β)
end

###############################################################################
# Spatial-chain benchmark cases
###############################################################################

function bench_spatial_migration_backward(s::SpatialBenchSetup)
    stage = s.move
    V_end = s.V_end
    env   = s.env
    buf   = s.buffers[2]
    return @benchmark backward!($stage, $V_end, $env, $buf)
end

function bench_spatial_migration_backward_ref(s::SpatialBenchSetup)
    V_end = s.V_end
    C = s.C
    ε = SPATIAL_EPSILON
    n_w, n_z, n_l = size(V_end)
    V_pre = similar(V_end)
    prob  = zeros(Float64, n_w, n_z, n_l, n_l)
    return @benchmark ref_migration_backward!($V_pre, $prob, $V_end, $C, $ε)
end

function bench_spatial_cs_backward(s::SpatialBenchSetup)
    stage = s.savings
    V_end = s.V_end
    env   = s.env
    buf   = s.buffers[4]
    return @benchmark backward!($stage, $V_end, $env, $buf)
end

function bench_spatial_cs_backward_ref(s::SpatialBenchSetup)
    V_end = s.V_end
    wgrid = s.wgrid
    V_pre = similar(V_end)
    policy = zeros(Int, size(V_end))
    β = BETA
    return @benchmark ref_cs_backward_3d!($V_pre, $policy, $V_end, $wgrid, $β)
end

# Full spatial chain backward+forward — one outer-loop step cost.
function bench_spatial_chain_pass(s::SpatialBenchSetup)
    chain   = s.shock ∘ₛ s.move ∘ₛ s.receipt ∘ₛ s.savings
    buffers = allocate(chain)
    V_end = s.V_end
    Λ_pre = s.Λ_pre
    env   = s.env
    backward!(chain, V_end, env, buffers)
    forward!(chain, Λ_pre, buffers)
    return @benchmark begin
        backward!($chain, $V_end, $env, $buffers)
        forward!($chain, $Λ_pre, $buffers)
    end
end


# Full chain backward then forward (no fixed-point iteration — just one
# pass each). This is what an outer-loop step costs.
function bench_chain_pass(s::BenchSetup)
    chain   = s.shock ∘ₛ s.receipt ∘ₛ s.savings
    buffers = allocate(chain)
    V_end = s.V_end
    Λ_pre = s.Λ_pre
    env   = s.env
    # Warm the chain so kernels are populated.
    backward!(chain, V_end, env, buffers)
    forward!(chain, Λ_pre, buffers)
    return @benchmark begin
        backward!($chain, $V_end, $env, $buffers)
        forward!($chain, $Λ_pre, $buffers)
    end
end

# Hand-coded reference chain pass: do the three stages back-to-back as
# explicit operations on fixed buffers.
function bench_chain_pass_ref(s::BenchSetup)
    n_w = length(s.wgrid)
    n_z = size(s.V_end, 2)
    V_end = s.V_end
    Λ_pre = s.Λ_pre
    wgrid = s.wgrid
    wpost = s.wpost
    P_y   = s.P_y
    P_y_T = Matrix(P_y')
    β     = BETA

    # Allocate everything outside the closure.
    V_post_cs   = similar(V_end)
    V_post_wc   = similar(V_end)
    V_pre_shock = similar(V_end)
    policy      = zeros(Int, size(V_end))

    Λ_post_shock = similar(Λ_pre)
    Λ_post_wc    = similar(Λ_pre)
    Λ_post_cs    = similar(Λ_pre)

    P_perm_in  = Matrix{Float64}(undef, n_z, n_w)
    P_perm_out = Matrix{Float64}(undef, n_z, n_w)

    return @benchmark begin
        # Backward: savings → wealthchange → markov.
        # The chain is shock ∘ savings ∘ ..., applied right-to-left in
        # backward: savings.backward gets the terminal V_end, then
        # wealthchange.backward, then shock.backward.
        ref_cs_backward!($V_post_wc, $policy, $V_end, $wgrid, $β)
        ref_wealthchange_backward!($V_post_cs, $V_post_wc, $wgrid, $wpost, Val(:linear))
        ref_markov_backward!($V_pre_shock, $V_post_cs, $P_y, $P_perm_in, $P_perm_out)
        # Forward: shock → wealthchange → savings.
        ref_markov_forward!($Λ_post_shock, $Λ_pre, $P_y_T, $P_perm_in, $P_perm_out)
        ref_wealthchange_forward!($Λ_post_wc, $Λ_post_shock, $wgrid, $wpost)
        # The CS forward is a degenerate-policy push; the cleanest hand
        # version is a per-cell scatter.
        fill!($Λ_post_cs, 0.0)
        @inbounds for zi in 1:size($Λ_post_wc, 2)
            for ki in 1:size($Λ_post_wc, 1)
                a = $policy[ki, zi]
                $Λ_post_cs[a, zi] += $Λ_post_wc[ki, zi]
            end
        end
    end
end


###############################################################################
# Reporting
###############################################################################

format_time(ns) = ns < 1000   ? @sprintf("%5.0f ns", ns)         :
                  ns < 1e6    ? @sprintf("%5.2f μs", ns/1e3)     :
                  ns < 1e9    ? @sprintf("%5.2f ms", ns/1e6)     :
                                @sprintf("%5.2f s",  ns/1e9)

function present(label, t_lib, t_ref)
    ratio = t_lib / t_ref
    @printf("  %-32s  lib %s  ref %s  ratio %.2fx\n",
            label, format_time(t_lib), format_time(t_ref), ratio)
    return ratio
end


###############################################################################
# Main
###############################################################################

function run_suite(label::String, s::BenchSetup)
    println("== ", label, " — N_w=", length(s.wgrid), " N_z=", size(s.V_end, 2), " ==")
    results = Pair{String, NTuple{2, Float64}}[]

    push!(results, "MarkovStage backward" =>
        (minimum(bench_markov_backward(s)).time,
         minimum(bench_markov_backward_ref(s)).time))
    push!(results, "MarkovStage forward" =>
        (minimum(bench_markov_forward(s)).time,
         minimum(bench_markov_forward_ref(s)).time))
    push!(results, "WealthChangeStage backward" =>
        (minimum(bench_wealthchange_backward(s)).time,
         minimum(bench_wealthchange_backward_ref(s)).time))
    push!(results, "WealthChangeStage forward" =>
        (minimum(bench_wealthchange_forward(s)).time,
         minimum(bench_wealthchange_forward_ref(s)).time))
    push!(results, "ConsumptionSavingsStage backward" =>
        (minimum(bench_cs_backward(s)).time,
         minimum(bench_cs_backward_ref(s)).time))
    push!(results, "Chain backward+forward" =>
        (minimum(bench_chain_pass(s)).time,
         minimum(bench_chain_pass_ref(s)).time))

    for (label, (t_lib, t_ref)) in results
        present(label, t_lib, t_ref)
    end
    return results
end


function run_spatial_suite(label::String, s::SpatialBenchSetup)
    n_w, n_z, n_l = size(s.V_end)
    println("== ", label, " — N_w=", n_w, " N_z=", n_z, " N_l=", n_l, " ==")
    results = Pair{String, NTuple{2, Float64}}[]

    push!(results, "MigrationStage backward" =>
        (minimum(bench_spatial_migration_backward(s)).time,
         minimum(bench_spatial_migration_backward_ref(s)).time))
    push!(results, "ConsumptionSavingsStage backward (3D)" =>
        (minimum(bench_spatial_cs_backward(s)).time,
         minimum(bench_spatial_cs_backward_ref(s)).time))

    # Chain pass: library-only (the full hand-coded reference would
    # duplicate the entire chain logic — not worth it for one number).
    chain_lib = minimum(bench_spatial_chain_pass(s)).time
    @printf("  %-32s  lib %s  (no ref baseline)\n",
            "Chain backward+forward", format_time(chain_lib))
    push!(results, "Chain backward+forward" => (chain_lib, NaN))

    for (label, (t_lib, t_ref)) in results
        isnan(t_ref) && continue
        present(label, t_lib, t_ref)
    end
    return results
end


function main()
    println("HouseholdStages benchmarks  —  Julia $(VERSION)")
    println("Comparing library stages vs. hand-coded reference kernels.")
    println()

    Random.seed!(0)

    large = haskey(ENV, "LARGE_BENCH") && ENV["LARGE_BENCH"] != "" &&
            lowercase(ENV["LARGE_BENCH"]) ∉ ("0", "false", "no")

    nw_aiy = large ? NW_BASE_LARGE   : NW_BASE
    nw_ks  = large ? NW_KS_LARGE     : NW_KS
    nw_sp  = large ? NW_SPATIAL_LARGE : NW_SPATIAL

    s_aiy = make_setup(nw_aiy, NZ_BASE, Y_GRID_BASE, P_Y_BASE)
    res_aiy = run_suite("Aiyagari size ($(nw_aiy), $(NZ_BASE))", s_aiy)

    println()
    s_ks = make_setup(nw_ks, NZ_KS, Y_GRID_KS, P_Y_KS)
    res_ks = run_suite("K-S size ($(nw_ks), $(NZ_KS))", s_ks)

    println()
    s_sp = make_spatial_setup(nw_sp, NZ_SPATIAL, NL_SPATIAL,
                              Y_GRID_BASE, P_Y_BASE)
    res_sp = run_spatial_suite("Spatial size ($(nw_sp), $(NZ_SPATIAL), $(NL_SPATIAL))", s_sp)

    println()
    println("Notes: 'ratio' is library-time / reference-time at min sample.")
    println("       Lower is better; 1.0x = parity, < 2x excellent, < 4x acceptable.")
    if large
        println("       LARGE_BENCH=1 — using N_w = $(nw_aiy) (example-driver size).")
    else
        println("       Default size — set LARGE_BENCH=1 to re-run at N_w = 400.")
    end

    return (aiyagari = res_aiy, ks = res_ks, spatial = res_sp)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
