####################################################################
# Aiyagari MIT Shock — Perfect-Foresight Transition (notebook)     #
####################################################################
#
# Self-contained driver: depends only on `HouseholdStages`. Same
# three-stage household chain as `aiyagari.jl`, augmented with a
# permanent TFP step `A_t = A_0 > 1` for all `t ≥ 1` consumed by
# `aiyagari_prices` during the transition. The walkthrough goes:
#
#   1. Parameters + state layout.
#   2. Household chain.
#   3. Pre-shock steady state at A = 1.
#   4. Post-shock steady state at A = A_0 (the new long-run).
#   5. MIT-shock transition (uses the library
#      `solve_transition_given_env_path!` helper).
#   6. Sequence-space Jacobian sketch.

using HouseholdStages
using Printf


# 1. Parameters and State Layout #
#--------------------------------#

@kwdef struct MITShockParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36
    δ :: Float64       = 0.08
    L :: Float64       = 1.0
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 100.0
end
Base.Broadcast.broadcastable(p::MITShockParams) = Ref(p)

_u_crra(c, ::Val{1}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)

function aiyagari_household(p::MITShockParams)
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid(p.w_min, p.w_max;
                                           length = p.N_w, spacing = :log)),
        StateAxis(:income, p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition = p.P_y)
    receipt = WealthChangeStage(layout;
        wealth_post = (cell; env) -> (1 + env.r) * cell.wealth + env.w * cell.income,
        wealth_axis = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
    return define_moments!(shock ∘ receipt ∘ savings;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end

function aiyagari_prices(K::Real, A::Real, p::MITShockParams)
    (; α, δ, L) = p
    r = α * A * (K / L)^(α - 1) - δ
    w = (1 - α) * A * (K / L)^α
    return (; r, w)
end

# Permanent step: A_t = A_0 for all t ≥ 1.
tfp_path(T::Int; A_0::Float64 = 1.05) = fill(A_0, T)

p = MITShockParams()
@printf "Calibration: β = %.3f, σ = %.2f, α = %.2f, δ = %.2f\n" p.β p.σ p.α p.δ


# 2. Household Stage Chain #
#--------------------------#

hh   = aiyagari_household(p)
dims = layout_size(input_layout(hh))

@printf "Layout: wealth %d × income %d = %d cells\n" dims[1] dims[2] prod(dims)


# 3. Pre-shock Steady State (A = 1) — Warm Start #
#------------------------------------------------#

# Tatonnement on K at A = 1. The library's
# `solve_steady_state_given_env!` does the per-K inner work
# (V backward to fixed point + Λ forward to stationarity); the outer
# loop is rolled here.

function solve_ss(hh, p; A::Float64, K_init::Float64,
                  update_speed = 0.05, rtol = 2e-2, max_iter = 500)
    K     = K_init
    V, Λ  = nothing, nothing
    K_err = Inf
    iters = 0
    while iters < max_iter
        env  = (; K, aiyagari_prices(K, A, p)...)
        res  = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        V, Λ = res.V, res.Λ
        K_supplied = compute_moments(hh, Λ, env).K_supplied
        K_err = abs(K_supplied - K) / K
        iters += 1
        K_err <= rtol && break
        K += update_speed * (K_supplied - K)
    end
    (; r, w) = aiyagari_prices(K, A, p)
    return (; K, V, Λ, r, w, iters)
end

println("Solving pre-shock steady state (A = 1.0)…")
ss_pre = solve_ss(hh, p; A = 1.0, K_init = 5.0)
@printf "  K_ss_pre = %.4f, r = %.4f, w = %.4f  (converged in %d iters)\n" ss_pre.K ss_pre.r ss_pre.w ss_pre.iters


# 4. Post-shock Steady State (A = A_0) — New Long-Run #
#-----------------------------------------------------#

println("Solving post-shock steady state (A = 1.05)…")
A_0     = 1.05
ss_post = solve_ss(hh, p; A = A_0, K_init = ss_pre.K)
@printf "  K_ss_post = %.4f, r = %.4f, w = %.4f  (converged in %d iters)\n" ss_post.K ss_post.r ss_post.w ss_post.iters


# 5. MIT-Shock Transition — Library Driver #
#------------------------------------------#

# Perfect-foresight transition under a one-time unanticipated permanent
# TFP step `A_0 = 1.05`. Inside the K-path tatonnement, each candidate
# `K_path` defines an `env_path`; we hand that to the library's
# `solve_transition_given_env_path!`, which allocates per-period chains
# sharing the Spec, runs a backward sweep (terminal `V_T = V_ss_post`)
# followed by a forward sweep (initial `Λ_0 = Λ_ss_pre`), and returns
# the V/Λ paths plus per-period moments. The outer loop reads off
# `K_supplied[t]` from `tr.moments_path` and damped-updates `K_path`.

T       = 100
A_path  = tfp_path(T; A_0 = A_0)
damping = 0.2
tol_tr  = 1e-3
maxiter = 200

K_ss_pre,  V_ss_pre,  Λ_ss_pre  = ss_pre.K,  ss_pre.V,  ss_pre.Λ
K_ss_post, V_ss_post, Λ_ss_post = ss_post.K, ss_post.V, ss_post.Λ

K_path      = collect(range(K_ss_pre, K_ss_post; length = T))
K_supplied  = zeros(T)
res_history = Float64[]
tr_iters    = 0
res         = Inf

println("\nSolving MIT-shock transition (permanent step A = $(A_0))…")
@time begin
    while tr_iters < maxiter
        env_path = [(; K = K_path[t], aiyagari_prices(K_path[t], A_path[t], p)...)
                    for t in 1:T]
        tr = solve_transition_given_env_path!(hh, env_path;
                                              Λ_0 = Λ_ss_pre, V_T = V_ss_post)
        for t in 1:T
            K_supplied[t] = tr.moments_path[t].K_supplied
        end

        global res = maximum(abs, K_supplied .- K_path)
        push!(res_history, res)
        global tr_iters += 1

        res <= tol_tr && break

        K_path .= (1 - damping) .* K_path .+ damping .* K_supplied
    end
end

tr_converged = res <= tol_tr
println(tr_converged ? "Converged in $tr_iters outer iterations." :
                        "DID NOT CONVERGE in $tr_iters outer iterations.")
@printf "  K_ss_pre        = %.4f\n" K_ss_pre
@printf "  K_ss_post       = %.4f\n" K_ss_post
@printf "  K[1]   (impact) = %.4f  (Δ from pre = %+0.4f)\n" K_path[1] (K_path[1] - K_ss_pre)
@printf "  K[5]            = %.4f\n" K_path[5]
@printf "  K[20]           = %.4f\n" K_path[20]
@printf "  K[50]           = %.4f\n" K_path[50]
@printf "  K[100] (≈end)   = %.4f  (Δ from post = %+0.4f)\n" K_path[100] (K_path[100] - K_ss_post)
println("  Residual history (last 5): ",
        round.(res_history[max(1, end - 4):end]; digits = 6))


# 6. Sequence-Space-Jacobian Demo #
#--------------------------------#

# Uses the pre-shock SS kernels (re-seated at the SS env) and a unit
# integrand `cell -> cell.wealth`. Assembles a 30×30 fake-news matrix.

println("\nSequence-space-Jacobian sketch at the pre-shock steady state (T_horizon = 30)…")
ss_env = (; K = K_ss_pre, aiyagari_prices(K_ss_pre, 1.0, p)...)
backward!(hh, V_ss_pre, ss_env)
forward!(hh,  Λ_ss_pre)

T_horizon = 30
𝓔 = expectation_vectors(hh, cell -> cell.wealth, T_horizon)
@printf "  produced %d expectation arrays of shape %s\n" length(𝓔) string(size(𝓔[1]))
@printf "  ⟨𝓔[1], Λ_ss⟩ = %.4f  (≈ K_ss_pre = %.4f)\n" sum(𝓔[1] .* Λ_ss_pre) K_ss_pre

curlyY = ones(T_horizon)
curlyD = [zeros(Float64, dims...) for _ in 1:T_horizon]
mid_w_idx = div(dims[1], 2)
for t in 1:T_horizon
    curlyD[t][mid_w_idx, 1]   =  1.0
    curlyD[t][mid_w_idx, end] = -1.0
end
F = build_F(curlyY, curlyD, 𝓔[2:end])
J = J_from_F(F)
@printf "  F : %s ; J : %s\n" string(size(F)) string(size(J))
@printf "  J[0, 0] = %+.6f, J[1, 0] = %+.6f, J[2, 0] = %+.6f\n" J[1, 1] J[2, 1] J[3, 1]

println("\nDone.")
