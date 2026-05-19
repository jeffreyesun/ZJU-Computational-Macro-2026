# Spatial Aiyagari — two locations with migration

Smallest spatial extension that exercises:
- A `categorical` axis with symbol-valued levels (`:home`, `:abroad`).
- The dedicated `MigrationStage` stage acting on that location axis
  (cost matrix + Gumbel scale ε — no user closure).
- The L03/L04 three-stage decomposition of the savings problem
  (`IncomeReceipt ∘ₛ ConsumptionSavingsStage`) sitting underneath the
  migration stage.
- Per-location moments via integrand closures that read
  `cell.location`.
- Two-dimensional outer-loop market clearing on
  `(K_home, K_abroad)`.

## State space + chain

```julia
StateLayout(
    StateAxis(:wealth,   continuous_grid(exp_wealth_grid(0, 30, 400))),
    StateAxis(:income,   discrete_finite([0.6, 1.0, 1.4])),
    StateAxis(:location, categorical([:home, :abroad])),
)

chain = IncomeShock ∘ₛ MigrationStage ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavingsStage
```

Four stages. The income shock resolves first; households then choose
location (logit-smoothed, with a migration cost for any move);
location-specific prices flow through `IncomeReceipt` (the
deterministic `b ↦ (1 + r_loc) b + w_loc y`); finally
`ConsumptionSavingsStage` picks next-period wealth on the grid with implicit
budget `c = b_in - b_end`.

The wealth grid is exponentially spaced (`exp_wealth_grid`) for the
same reason as the other examples: `WealthChangeStage.backward` linearly
interpolates V past the top knot for cells where
`(1 + r_loc) b + w_loc y > w_max`, and a uniform grid amplifies V per
pass and breaks the Bellman contraction.

## The `MigrationStage` stage

The 2026-05-16 overnight added a dedicated `MigrationStage` stage that
replaces the prior `LogitChoiceStage`-with-flow-payoff-closure pattern.
The API is:

```julia
MigrationStage(layout;
    location_axis  = :location,
    migration_cost = [0.0 0.5;           # (n_loc, n_loc), origin → destination
                      0.5 0.0],
    ε              = 5.0,                # Gumbel scale (literal or Param)
)
```

`migration_cost[i, j]` is the cost of moving from location `i` to `j`
(diagonal is typically zero — staying put is free). `ε` is the
Gumbel taste-shock scale; the kernel computes

```
π(j | i, s) ∝ exp((-C[i, j] + V_end[j, s]) / ε)
```

and backward yields `V_pre[..., i, ...] = ε log Σ_j π(j | i, s) · ε`
(the standard log-sum-exp). No user closure is needed for the cost;
shape is checked at construction; the cost matrix and ε flow through
`with_eltype` as shared static data for ForwardDiff lifts.

## Per-location moments

`lift_moments` supports integrands that read cell coordinates, which
is how per-location capital is computed without a dedicated
"per-axis" moment-spec kwarg:

```julia
K_home = at_end(
    integrand = (cell; env) -> cell.location == :home ? cell.wealth : 0.0,
    reduce    = sum,
)
```

Four moments: `K_home`, `K_abroad`, `pop_home`, `pop_abroad`.

## Outer loop

Damped Picard on the pair `(K_home, K_abroad)`. The model is symmetric
under the baseline calibration (`A_home = A_abroad`); equilibrium is
`K_home = K_abroad ≈ 2.8305` with population split 50/50.

## Calibration

Equal TFP across locations is the baseline. **A non-zero productivity
gradient produces strong migration-driven oscillations** in damped
Picard: when one location has slightly higher prices, mass flows
there; that depresses prices and shifts mass back. We observed
period-3 orbits in early experiments with `A_home/A_abroad =
1.05/0.95` even with damping = 0.1, consistent with chaotic 1-D
dynamics. Anderson acceleration or Newton on the residual would
converge in those regimes; this example uses damped Picard for
simplicity and stays in the symmetric regime.

The logit `ε_logit = 5.0` is high — close to uniform conditional on
`migration_cost`. Lowering it makes the household chain more
responsive to small price gradients — economically richer,
computationally harder.

## Run

```bash
julia --project=. HouseholdStages/examples/spatial/steady_state.jl
```

The four-section walkthrough lives in
`HouseholdStages/examples/notebooks/spatial.jl`.

## Expected result

```
K_home    = K_abroad = 2.8305
K_total              = 5.661
pop_home  = pop_abroad = 0.5
r         ≈ 0.038 (both locations)
w         ≈ 1.196 (both locations)
```

Three outer iterations at `N_w = 400`. Total capital
`K_home + K_abroad = 5.66` corresponds to the half-population
Aiyagari per location (each location has `L_each = L / 2 = 0.5`).
Single-location Aiyagari at `L = 1` lands `K = 5.6852` on the same
post-2026-05-15 3-stage chain, so this two-location version with
equal TFP lands just slightly below that — the small downward shift
comes from migration-induced mass mixing each period, which keeps the
ergodic wealth distribution slightly less concentrated near the
borrowing constraint than the single-location problem.

## Files

| File | Purpose |
|---|---|
| `model.jl` | Stages (incl. `MigrationStage`), prices, household chain |
| `steady_state.jl` | Damped Picard on `(K_home, K_abroad)` |
| `../notebooks/spatial.jl` | Four-section walkthrough driver |
