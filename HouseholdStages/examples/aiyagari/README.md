# Aiyagari (1994) — `HouseholdStages` tutorial

The smallest end-to-end exercise of the package: a heterogeneous-agent
steady state solved by tatonnement on aggregate capital `K`, with the
household layer decomposed into a three-stage chain.

The within-period problem is

```
IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings
```

a Markov draw on the income axis, a deterministic wealth update
`b ↦ (1+r) b + w y`, and a hard-argmax choice of `b_end` on the
wealth grid. The outer loop is plain Julia — no `AbstractBlock`, no
`EquilibriumProblem` — running damped Picard (tatonnement) on `K`.

## Step 1 — Parameters

```julia
@kwdef struct AiyagariParams
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
Base.Broadcast.broadcastable(p::AiyagariParams) = Ref(p)
```

`Base.Broadcast.broadcastable(::AiyagariParams) = Ref(p)` lets the
`@kwdef` struct flow through `.`-broadcasts as a scalar — the stages'
closures see `p` whole, never a slice.

## Step 2 — Household state layout

The state is `(wealth, income)` with wealth on an exponentially-spaced
grid and income on a 3-point discrete-finite axis:

```julia
function exp_wealth_grid(lo, hi, n; shift = 1.0)
    return [exp(t) * shift - shift + lo
            for t in range(0.0, log((hi - lo + shift) / shift); length = n)]
end

function aiyagari_layout(p = aiyagari_params)
    return StateLayout(
        StateAxis(:wealth, continuous_grid(exp_wealth_grid(p.w_min, p.w_max, p.N_w))),
        StateAxis(:income, discrete_finite(p.y_grid)),
    )
end
```

The exponential grid is dense near `w_min` (where the borrowing
constraint binds) and coarse at the top. `WealthChange.backward`
linearly *extrapolates* V past the top knot; an evenly spaced grid
amplifies V by `extrap_distance / top_step` each pass and breaks the
Bellman contraction. The exponential transform makes the top span far
enough that this never bites at the calibration's `(1+r) w + w·y`.

## Step 3 — Define the three stages

**Income shock** is the canonical V/θ-independent stage: a Markov
transition along the named axis. Backward applies `Pᵀ`, forward
applies `P`, kernel is the transition matrix itself.

```julia
function aiyagari_income_shock(layout, p = aiyagari_params)
    return MarkovAlong(layout; axis = :income, transition = p.P_y)
end
```

**Income receipt** is the deterministic wealth update. The closure
`wp(cell; env)` receives the cell whole (a `NamedTuple` over the state
axes) and `env` as a `Ref` — unwrap with `env[]` before field access.
`closure_deps = (:r, :w)` declares which fields of `env` the closure
reads (used by the dependency checker and Jacobian lifts).

```julia
function aiyagari_income_receipt(layout, p = aiyagari_params)
    function wp(cell; env)
        e = env[]
        return (1 + e.r) * cell.wealth + e.w * cell.income
    end
    return WealthChange(layout;
        wealth_post  = wp,
        wealth_axis  = :wealth,
        closure_deps = (:r, :w),
    )
end
```

**Consumption-savings** picks `b_end` on the wealth grid; implied
consumption is `c = b_in − b_end`. The `monotone_search =
:divide_conquer` opt-in switches the inner argmax to the
divide-and-conquer walk (strictly `O(n_w log n_w)` per slice instead
of the sequential walk's `O(n_w + Σ bound_widths)`). The D&C path
requires the savings policy to be non-decreasing in input wealth
(non-negative MPS); concave `u` + linear budget guarantees this.
If your payoff isn't concave-and-linear, leave it at
`:sequential` — see the `ConsumptionSavings` docstring for the
correctness caveat.

```julia
function aiyagari_consumption_savings(layout, p = aiyagari_params)
    return ConsumptionSavings(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )
end
```

CRRA utility with the `Val{σ}` dispatch fires log-utility for `σ = 1`
and the power form otherwise:

```julia
_u_crra(c, ::Val{1}) = log(c)
_u_crra(c, ::Val{σ}) where σ = (c^(1 - σ)) / (1 - σ)
u_crra(c, valσ::Val) = c < 0 ? -Inf : _u_crra(c, valσ)
```

## Step 4 — Compose with `∘ₛ`, attach the moment

`∘ₛ` builds a `StageChain` (allocation-free `@generated` unroll under
the hood). `lift_moments` wraps the chain with a moment-emission
closure: `K_supplied = at_end(integrand = :wealth, reduce = sum)`
says "after one forward pass, integrate `wealth` against the
terminal Λ" — and `compute_moments(hh, env)` reads it off.

```julia
function aiyagari_household(p = aiyagari_params)
    layout  = aiyagari_layout(p)
    chain   = aiyagari_income_shock(layout, p) ∘ₛ
              aiyagari_income_receipt(layout, p) ∘ₛ
              aiyagari_consumption_savings(layout, p)
    return lift_moments(chain;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end
```

## Step 5 — Solve for steady state

The outer loop is damped Picard on `K`:

```julia
function aiyagari_steady_state(p = aiyagari_params;
                               K_init = 5.0, update_speed = 0.05,
                               rtol = 2e-2, max_iter = 500)
    hh = aiyagari_household(p)
    caches, scratches = allocate(hh)

    dims = layout_size(aiyagari_layout(p))
    V    = zeros(Float64, dims...)
    Λ    = fill(1.0 / prod(dims), dims...)

    K = K_init
    K_err = Inf
    iters = 0
    while abs(K_err) > rtol
        (; r, w) = aiyagari_prices(K, p)
        env = (; K, r, w)
        V   = aiyagari_vfi!(hh, env, V, caches, scratches).V
        Λ   = aiyagari_lambda!(hh, Λ, caches, scratches).Λ
        (; K_supplied) = compute_moments(hh, env)
        K_err = (K_supplied - K) / K
        K += update_speed * K_err * K
        iters += 1
        iters == max_iter && break
    end
    return (; K, r, w, V, Λ)
end
```

`aiyagari_vfi!` runs `backward!` until V converges.
`aiyagari_lambda!` runs `forward!` until Λ stabilizes.

## Step 6 — Inspect

The default calibration converges to:

```
K = 5.6852
r = 3.7 %
w  = 1.20
Σ Λ = 1.0
```

in ~18 outer iterations. Representative-agent baseline at this
calibration has `K_RA ≈ 5.0`; the lift to `K = 5.69` is the standard
Aiyagari precautionary-savings effect.

Drop-in plots: marginal `K_supplied` along the wealth axis (the
savings curve), the stationary Λ distribution, or the policy
`b_end(b_in, y)` recovered from `caches.savings.policy`. The
notebook `../notebooks/aiyagari.jl` walks through these
interactively.

## Run

```
julia --project=. HouseholdStages/examples/aiyagari/steady_state.jl
```

About 5 seconds on a single CPU core at `N_w = 400`.

## Files

- `model.jl`         — parameters, layout, stage constructors, prices.
- `steady_state.jl`  — outer K tatonnement + inner VFI + inner Λ.
- `README.md`        — this tutorial.
