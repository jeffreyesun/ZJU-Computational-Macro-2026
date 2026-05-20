# Krusell-Smith (1998) — deterministic-aggregate steady state

The K-S household block at constant TFP `A = 1.0`. Same chain shape
as Aiyagari, specialised to the K-S two-state employed/unemployed
income process.

## What's K-S-specific

- **Two-state income.** `y_grid = [0.07, 1.0]`; employed/unemployed
  with `P_y = [0.6 0.4; 0.05 0.95]` (stationary unemployment ≈ 11 %).
  The unemployed income `y_unemp = 0.07` is the canonical K-S
  calibration — strictly positive so the `b = 0` corner stays
  feasible under log utility.
- **Log utility.** `γ = 1` via the explicit `Val{1.0}` dispatch (note
  `Val(1.0)` lands in the Float branch of the dispatch, not the Int
  branch — both are wired up in `model.jl`).
- **Wider wealth grid.** `w_max = 200` (vs. Aiyagari's 100) because
  K-S sits at the impatience watershed `β(1+r) ≈ 1`, where households
  save aggressively in response to small changes in `r`.
- **Effective labour computed from the stationary income
  distribution.** `ks_effective_labor(P_y, y_grid)` solves
  `π' P = π'`, `Σ π = 1` and returns `Σ π_i y_i`. The Cobb-Douglas
  wage and rental rate read this through `ks_prices`.

## The chain

```julia
function ks_household(p = ks_params)
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
        utility         = (cell, c; env) -> u_crra(c, Val(p.γ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end
```

## Outer loop — tatonnement, not bisection

Damped tatonnement on `K` with `K_init = 13.6`, `update_speed = 0.01`,
`rtol = 5e-2`. The example uses tatonnement rather than bisection
because the three-stage chain's `WealthChangeStage.backward` does not
survive bisection's extreme-K probes: at small `K`, `r` rises past
0.3 and the linear V-extrapolation past `w_max` amplifies V faster
than `1 / β`, breaking the Bellman contraction. Tatonnement walks
near the candidate K throughout, and the linear extrapolation stays
benign.

## Calibration and result

```
K          = 12.8791
r          = 4.04 %
w          = 1.6703
β (1 + r)  = 0.9988
ΣΛ         = 1.0
```

in 24 outer iterations.

## Why the residual floor is wider than Aiyagari's

The hard-argmax `ConsumptionSavingsStage` produces a step-function
`K_supplied(K)` curve, and K-S sits right at a sharp policy switch:
a one-grid-step change in the chosen `b_end` moves `K_supplied` by
≈ 20 %, so the tatonnement floor at `N_w = 400` is ≈ 3–5 % rather
than Aiyagari's 1–2 %. `rtol = 5e-2` accepts that floor. Tightening
would require either further grid refinement or replacing the hard
argmax with a smoothed `LogitChoiceStage`-based savings policy.

## Stochastic-aggregate K-S

The full Krusell-Smith problem with stochastic aggregate TFP requires
training a neural-network value-function approximator on the pooled
Bellman residual. That direction is out of scope for this library —
`HouseholdStages` deliberately keeps the household chain at a single
env and leaves aggregate-shock machinery to the consumer.

## Run

```bash
julia --project=. examples/krusell_smith/steady_state.jl
```

About 10–15 seconds at `N_w = 400`.

## Files

- `model.jl` — parameters, layout, stage constructors,
  `ks_effective_labor`, `ks_prices`.
- `steady_state.jl` — outer tatonnement.
- `../notebooks/krusell_smith.jl` — Pluto-style four-section
  walkthrough.
