# Aiyagari (1994) — steady state

The smallest end-to-end exercise of `HouseholdStages`: a
heterogeneous-agent steady state solved by damped tatonnement on
aggregate capital `K`, with the household layer decomposed into a
three-stage chain.

## The chain

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

- **`IncomeShock`** — `MarkovStage` on the income axis. The K-operator
  is just `P_y`; backward applies `P_y'`, forward applies `P_y`.
- **`IncomeReceipt`** — `WealthChangeStage` with `wealth_post = (cell;
  env) -> (1 + env.r) * cell.wealth + env.w * cell.income`. Backward
  interpolates `V_end` linearly along the wealth axis at each cell's
  post-receipt wealth; forward redistributes mass to the wealth grid
  via share-weighted accumulation.
- **`ConsumptionSavingsStage`** — pick `b_end` on the wealth grid;
  implicit budget `c = b_in − b_end`; CRRA utility. Inner argmax via
  the divide-and-conquer monotone-policy walk (`O(n_w log n_w)` per
  slice; concave `u` + linear budget gives the required non-negative
  MPS).

```julia
function aiyagari_household(p = AiyagariParams())
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

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end
```

## Why the wealth grid is log-spaced

`continuous_grid(lo, hi; spacing = :log)` gives a grid dense near
zero (where the borrowing constraint binds and policies are highly
nonlinear) and coarse at the top. `WealthChangeStage.backward`
interpolates `V_end` linearly — under `:linear` extrapolation past
the top knot, the V-amplification factor is
`extrap_distance / top_step`, and on a uniform grid this can exceed
`1 / β` during off-equilibrium probes and break the Bellman
contraction. The log grid makes the top span wide enough that
extrapolation doesn't bite at the calibration's
`(1 + r) w + w · y`.

## The outer loop

Damped tatonnement on `K`:

```julia
while iterations < max_iter
    env = (; K, aiyagari_prices(K, p)...)
    res = isnothing(V) ?
        solve_steady_state_given_env!(hh, env) :
        solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
    (; V, Λ) = res

    K_supplied = res.moments.K_supplied
    K_err = abs(K_supplied - K) / K
    K_err <= rtol && break

    K += update_speed * (K_supplied - K)
    iterations += 1
end
```

The library handles the per-K inner work
(`solve_steady_state_given_env!` runs V backward to a fixed point
then Λ forward to its stationary distribution, then evaluates
attached moments). Closing the model — the tatonnement update,
calibration, residual semantics — lives here in the example folder.

The first call uses default `V_init = zero`, `Λ_init = uniform`;
subsequent calls warm-start from the previous solution.

## Calibration and result

Default `AiyagariParams`: β = 0.96, σ = 1.5, α = 0.36, δ = 0.08;
3-state income with `y_grid = [0.6, 1.0, 1.4]` and a symmetric
sticky transition matrix; `N_w = 400` log-spaced wealth points on
`[0, 100]`.

```
K       = 5.6847
r       = 3.84 %
w       = 1.1964
ΣΛ      = 1.0
```

in 18 outer iterations (`update_speed = 0.05`, `rtol = 2e-2`). The
representative-agent baseline at this calibration has `K_RA ≈ 5.0`;
the lift to `K = 5.69` is the standard Aiyagari precautionary-savings
effect.

## Discretisation floor

Hard-argmax `ConsumptionSavingsStage` produces a step function in
`K_supplied(K)`. The floor on the relative residual at `N_w = 400` is
~1–2%; default `rtol = 2e-2` accepts the floor. Tighten by refining
the wealth grid further or by replacing the hard argmax with a
smoothed `LogitChoiceStage`-based savings policy (would also enable
finite-difference cross-checks against SSJ Jacobians at small ε).

## Run

```bash
julia --project=. examples/aiyagari/steady_state.jl
```

About 5 seconds at `N_w = 400` after the first compilation pass.

## Files

- `model.jl` — parameters, layout, stage constructors, prices.
- `steady_state.jl` — outer tatonnement, inner work via
  `solve_steady_state_given_env!`.
- `../notebooks/aiyagari.jl` — Pluto-style four-section walkthrough.
