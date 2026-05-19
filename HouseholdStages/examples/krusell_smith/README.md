# Krusell-Smith (1998) — deterministic-aggregate steady state

This example covers the **deterministic-aggregate** K-S steady state
at constant TFP `A = 1.0`. Same chain shape as Aiyagari, with the K-S
two-state employed / unemployed income process.

## Household stage (refactored 2026-05-15)

```julia
chain = MarkovStage(:income) ∘ₛ WealthChangeStage(b → (1+r)b + w y) ∘ₛ ConsumptionSavingsStage
hh    = lift_moments(chain; K_supplied = at_end(integrand = :wealth, reduce = sum))
```

The canonical L03 / L04 three-stage decomposition. `IncomeShock`
resolves the Markov draw on `:income`. `IncomeReceipt` is a
deterministic wealth update `b ↦ (1+r) b + w y`. `ConsumptionSavingsStage`
chooses next-period wealth on the wealth grid with implicit budget
`c = b_in - b_end`.

Two-state income process: employed (`y = 1.0`) / unemployed
(`y_unemp = 0.07`). The canonical K-S unemployed-income calibration —
strictly positive so the `b = 0` corner remains feasible under log
utility. Log utility (γ = 1).

## Calibration

- `β = 0.96` (annual-style), `α = 0.36`, `δ = 0.025`.
- `P_y = [0.6 0.4; 0.05 0.95]` (stationary unemployment ≈ 11%).
- Wealth grid: 100 points, exponentially spaced over `[0, 200]`.

`w_max = 200` and `N_w = 100` are wider and finer than Aiyagari's
defaults because K-S sits at the impatience watershed `β(1+r) ≈ 1`,
where the household saves aggressively in response to small changes in
r. Without the wider top, `WealthChangeStage.backward`'s linear V-extrapolation
past `w_max` would amplify V each pass and break the Bellman contraction
during off-equilibrium tatonnement probes.

## Outer loop

Damped Picard (tatonnement) on `K`, with `K_init = 13.6` and
`update_speed = 0.01`. Identical structure to Aiyagari's tatonnement.

The example uses tatonnement rather than bisection because the 3-stage
chain's `WealthChangeStage.backward` does not survive bisection's extreme-K
probes: at small K, `r` rises to 0.3+ and the linear V-extrapolation's
amplification factor exceeds `1/β`, breaking the Bellman contraction.
Tatonnement stays near the candidate K throughout.

## Run

```bash
julia --project=. HouseholdStages/examples/krusell_smith/steady_state.jl
# or, with the four-section walkthrough:
julia --project=. HouseholdStages/examples/notebooks/krusell_smith.jl
```

## Expected result

```
K ≈ 12.88
r ≈ 0.0404
w ≈ 1.6703
β(1+r) ≈ 0.9988
```

(Updated 2026-05-17 after the `N_w` bump from 100 to 400 — the
finer grid shaves a discretisation step off `K_supplied(K)`'s
floor and the converged `K` lands lower than the
2026-05-15 `K = 13.46` baseline. The earlier number was the same
chain at `N_w = 100`.)

The hard-argmax `ConsumptionSavingsStage` produces a step-function
`K_supplied(K)` curve, and K-S sits right at the policy switch: a
one-grid-step change in policy moves `K_supplied` by ~20%, so the
tatonnement's residual floor is ~3-5% rather than Aiyagari's 1-2%.
`rtol = 0.05` accepts this floor. Tighten by refining the wealth grid
further or by replacing the hard argmax with a smoothed `LogitChoiceStage`
savings policy.

## Stochastic-aggregate K-S is dropped

The full K-S with stochastic aggregate TFP would need vnet training on
the Pooled Bellman residual. The `train_vnet` loop was scaffolded in
an earlier iteration but never implemented; it has been dropped from
the 2026-05-13 refactor. Scaffolding is preserved in
`_attic/CVIAYN_core_code/src/lifts/aggregate_state.jl`.
