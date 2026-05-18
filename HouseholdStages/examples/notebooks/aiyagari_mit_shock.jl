####################################################################
# Aiyagari MIT Shock — Perfect-Foresight Transition (notebook)     #
####################################################################
#
# Self-contained driver: depends only on `HouseholdStages`. Same
# three-stage household chain as `aiyagari.jl`, augmented with a
# deterministic AR(1) TFP path `A_t` consumed by `aiyagari_prices`
# during the transition. The walkthrough goes:
#
#   1. Parameters + state layout.
#   2. Household chain and workspace.
#   3. Deterministic steady state (warm start for the transition).
#   4. MIT-shock transition (rolled here, period-by-period).
#   5. Sequence-space Jacobian sketch.

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

function exp_wealth_grid(lo::Real, hi::Real, n::Int; shift::Real = 1.0)
    return [exp(t) * shift - shift + lo
            for t in range(0.0, log((hi - lo + shift) / shift); length = n)]
end

function aiyagari_layout(p::MITShockParams)
    return StateLayout(
        StateAxis(:wealth, continuous_grid(exp_wealth_grid(p.w_min, p.w_max, p.N_w))),
        StateAxis(:income, discrete_finite(p.y_grid)),
    )
end

_u_crra(c, ::Val{1}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)

function aiyagari_household(p::MITShockParams)
    layout = aiyagari_layout(p)
    shock  = MarkovAlong(layout; axis = :income, transition = p.P_y)
    receipt = WealthChange(layout;
        wealth_post  = (cell; env) -> (e = env[]; (1 + e.r) * cell.wealth + e.w * cell.income),
        wealth_axis  = :wealth,
        closure_deps = (:r, :w),
    )
    savings = ConsumptionSavings(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
    return lift_moments(shock ∘ₛ receipt ∘ₛ savings;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end

function aiyagari_prices(K::Real, A::Real, p::MITShockParams)
    (; α, δ, L) = p
    r = α * A * (K / L)^(α - 1) - δ
    w = (1 - α) * A * (K / L)^α
    return (; r, w)
end

function tfp_path(T::Int; A_0::Float64 = 1.05, ρ::Float64 = 0.85, A_ss::Float64 = 1.0)
    A = zeros(T)
    A[1] = A_0
    for t in 2:T
        A[t] = A_ss + ρ * (A[t-1] - A_ss)
    end
    return A
end

p      = MITShockParams()
layout = aiyagari_layout(p)
dims   = layout_size(layout)

@printf "Calibration: β = %.3f, σ = %.2f, α = %.2f, δ = %.2f\n" p.β p.σ p.α p.δ
@printf "Layout: wealth %d × income %d = %d cells\n" dims[1] dims[2] prod(dims)


# 2. Household Stage Chain and Workspace #
#----------------------------------------#

hh = aiyagari_household(p)
caches, scratches = allocate(hh)
V = zeros(Float64, dims...)
Λ = fill(1.0 / prod(dims), dims...)

@printf "Λ_init: %d cells, ΣΛ = %.6f\n" length(Λ) sum(Λ)


# 3. Steady State (warm start) #
#------------------------------#

# Tatonnement on K at A = 1. The library's
# `solve_steady_state_given_env!` does the per-K inner work
# (V backward to fixed point + Λ forward to stationarity); the outer
# loop is rolled here.

println("Solving deterministic steady state (A = 1.0)…")
K_ss          = 5.0
update_speed  = 0.05
rtol_ss       = 2e-2
K_err         = Inf
ss_iters      = 0
@time begin
    while ss_iters < 500
        env = (; K = K_ss, aiyagari_prices(K_ss, 1.0, p)...)
        info = solve_steady_state_given_env!(hh, env, V, Λ, caches, scratches)
        global V, Λ = info.V, info.Λ

        K_supplied = compute_moments(hh, env).K_supplied
        global K_err = abs(K_supplied - K_ss) / K_ss
        global ss_iters += 1

        K_err <= rtol_ss && break

        global K_ss += update_speed * (K_supplied - K_ss)
    end
end
(; r, w) = aiyagari_prices(K_ss, 1.0, p)
@printf "  K_ss = %.4f, r_ss = %.4f, w_ss = %.4f; converged in %d iters\n" K_ss r w ss_iters

V_ss = copy(V)
Λ_ss = copy(Λ)


# 4. MIT-Shock Transition #
#-------------------------#

# Perfect-foresight transition under a one-time unanticipated TFP
# impulse `A_0 = 1.05` with AR(1) decay `ρ = 0.85`. The transition
# driver is rolled here, period by period:
#   (a) backward sweep with terminal V_{T+1} = V_ss,
#   (b) forward sweep with initial Λ_1 = Λ_ss (re-seating per-stage
#       caches at each period-t env),
#   (c) damped tatonnement update on the K-path.

T       = 100
A_path  = tfp_path(T; A_0 = 1.05, ρ = 0.85, A_ss = 1.0)
damping = 0.2
tol_tr  = 1e-3
maxiter = 200

V_path  = [copy(V_ss) for _ in 1:(T + 1)]
Λ_path  = [zeros(Float64, dims...) for _ in 1:(T + 1)]
Λ_path[1] .= Λ_ss

K_t        = fill(K_ss, T)
K_supplied = zeros(T)
res_history = Float64[]
tr_iters   = 0
res        = Inf

println("\nSolving MIT-shock transition (T = $T, A_0 = 1.05, ρ = 0.85)…")
@time begin
    while tr_iters < maxiter
        # 4a. Backward sweep
        for t in T:-1:1
            env = (; K = K_t[t], aiyagari_prices(K_t[t], A_path[t], p)...)
            V_path[t] .= backward!(hh, V_path[t + 1], env, caches, scratches)
        end

        # 4b. Forward sweep — re-seat caches at each period-t env, then
        # push Λ and read off K_supplied.
        for t in 1:T
            env = (; K = K_t[t], aiyagari_prices(K_t[t], A_path[t], p)...)
            backward!(hh, V_path[t + 1], env, caches, scratches)
            Λ_path[t + 1] .= forward!(hh, Λ_path[t], caches, scratches)
            K_supplied[t]   = compute_moments(hh, env).K_supplied
        end

        # 4c. Residual + damped update
        global res = maximum(abs, K_supplied .- K_t)
        push!(res_history, res)
        global tr_iters += 1

        res <= tol_tr && break

        K_t .= (1 - damping) .* K_t .+ damping .* K_supplied
    end
end

tr_converged = res <= tol_tr
println(tr_converged ? "Converged in $tr_iters outer iterations." :
                        "DID NOT CONVERGE in $tr_iters outer iterations.")
@printf "  K_ss            = %.4f\n" K_ss
@printf "  K[1]   (impact) = %.4f  (Δ = %+0.4f)\n" K_t[1] (K_t[1] - K_ss)
@printf "  K[5]            = %.4f  (Δ = %+0.4f)\n" K_t[5] (K_t[5] - K_ss)
@printf "  K[20]           = %.4f  (Δ = %+0.4f)\n" K_t[20] (K_t[20] - K_ss)
@printf "  K[50]           = %.4f  (Δ = %+0.4f)\n" K_t[50] (K_t[50] - K_ss)
@printf "  K[100] (≈end)   = %.4f  (Δ = %+0.4f)\n" K_t[100] (K_t[100] - K_ss)
@printf "  A[1] / A[10] / A[50] = %.4f / %.4f / %.4f\n" A_path[1] A_path[10] A_path[50]
println("  Residual history (last 5): ",
        round.(res_history[max(1, end - 4):end]; digits = 6))


# 5. Sequence-Space-Jacobian Demo #
#--------------------------------#

# Uses the SS caches (re-seated at the SS env) and a unit integrand
# `cell -> cell.wealth`. Assembles a 30×30 fake-news matrix.

println("\nSequence-space-Jacobian sketch at the steady state (T_horizon = 30)…")
ss_env = (; K = K_ss, aiyagari_prices(K_ss, 1.0, p)...)
ssj_caches, ssj_scratches = allocate(hh)
backward!(hh, V_ss, ss_env, ssj_caches, ssj_scratches)
forward!(hh, Λ_ss, ssj_caches, ssj_scratches)

T_horizon = 30
𝓔 = expectation_vectors(hh, cell -> cell.wealth, T_horizon, ssj_caches, ssj_scratches)
@printf "  produced %d expectation arrays of shape %s\n" length(𝓔) string(size(𝓔[1]))
@printf "  ⟨𝓔[1], Λ_ss⟩ = %.4f  (≈ K_ss = %.4f)\n" sum(𝓔[1] .* Λ_ss) K_ss

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
