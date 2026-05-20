# Spatial Aiyagari — two locations with migration

The smallest spatial extension of the Aiyagari household block.
Exercises:

- A `categorical` axis with symbol-valued levels (`[:home, :abroad]`).
- The dedicated `MigrationStage` acting on that location axis
  (cost matrix + Gumbel scale ε — no user closure for the cost).
- The L03/L04 three-stage savings decomposition sitting underneath
  the migration stage.
- **Per-location moments via integrand closures** that read
  `cell.location`.
- A two-dimensional outer-loop market clearing on
  `(K_home, K_abroad)`.

## The chain

```
IncomeShock ∘ MigrationStage ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

Four stages. The income shock resolves first; households then choose
a destination (logit-smoothed with a symmetric migration cost);
location-specific prices flow through `IncomeReceipt` (the
deterministic `b ↦ (1 + r_loc) b + w_loc y`); finally
`ConsumptionSavingsStage` picks next-period wealth on the grid.

```julia
function spatial_household(p = params)
    layout = StateLayout(
        StateAxis(:wealth,   continuous_grid(p.w_min, p.w_max;
                                             length = p.N_w, spacing = :log)),
        StateAxis(:income,   p.y_grid),
        StateAxis(:location, categorical([:home, :abroad])),
    )

    shock   = MarkovStage(layout; axis = :income, transition = p.P_y)
    move    = MigrationStage(layout;
        location_axis  = :location,
        migration_cost = [0.0              p.migration_cost;
                          p.migration_cost 0.0],
        ε              = Param(p.ε_logit),
    )
    receipt = WealthChangeStage(layout;
        wealth_post = function (cell; env)
            r_loc = cell.location == :home ? env.r_home : env.r_abroad
            w_loc = cell.location == :home ? env.w_home : env.w_abroad
            return (1 + r_loc) * cell.wealth + w_loc * cell.income
        end,
        wealth_axis = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β               = p.β,
        utility         = (cell, c; env) -> u_crra(c, Val(p.σ)),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )

    hh = shock ∘ move ∘ receipt ∘ savings
    return define_moments!(hh;
        K_home     = at_end(integrand = (cell; env) -> cell.location == :home   ? cell.wealth : 0.0,
                            reduce = sum),
        K_abroad   = at_end(integrand = (cell; env) -> cell.location == :abroad ? cell.wealth : 0.0,
                            reduce = sum),
        pop_home   = at_end(integrand = (cell; env) -> cell.location == :home   ? 1.0 : 0.0,
                            reduce = sum),
        pop_abroad = at_end(integrand = (cell; env) -> cell.location == :abroad ? 1.0 : 0.0,
                            reduce = sum),
    )
end
```

## The `MigrationStage`

```julia
MigrationStage(layout;
    location_axis  = :location,
    migration_cost = [0.0 0.5; 0.5 0.0],   # (n_loc, n_loc), origin → destination
    ε              = Param(5.0),            # Gumbel scale
)
```

`migration_cost[i, j]` is the cost of moving from location `i` to
`j` (diagonal is zero — staying put is free). `ε` is the Gumbel
taste-shock scale; the kernel computes

```
π(j | i, s) ∝ exp((-C[i, j] + V_end[j, s]) / ε)
```

and backward yields the log-sum-exp value at each cell. The cost
matrix is stored on the Spec; the per-destination amenity field
(optional, see the Migration stage's docstring) is left at its
default `nothing` here. No user closure for the cost; shape is
checked at construction; the cost matrix flows through `with_eltype`
as shared static data for ForwardDiff lifts.

## Per-location moments via integrand closures

`lift_moments` (now `define_moments!`) supports integrands that read
cell coordinates — which is how per-location capital is computed
without a dedicated "per-axis" moment-spec kwarg:

```julia
K_home = at_end(
    integrand = (cell; env) -> cell.location == :home ? cell.wealth : 0.0,
    reduce    = sum,
)
```

Four moments total: `K_home`, `K_abroad`, `pop_home`, `pop_abroad`.

## Outer loop

Damped Picard on the pair `(K_home, K_abroad)`:

```julia
pr = spatial_prices(K_home, K_abroad, p)
env = (; K_home, K_abroad,
         pr.r_home, pr.w_home, pr.r_abroad, pr.w_abroad)
res = solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
m   = compute_moments(hh, res.Λ, env)

K_home   = (1 - d) * K_home   + d * m.K_home
K_abroad = (1 - d) * K_abroad + d * m.K_abroad
```

The library handles the per-pair inner work; the consumer rolls the
2-D damped Picard.

## Calibration

`A_home = A_abroad = 1.0` (symmetric baseline) and `ε_logit = 5.0`.
The high ε keeps the migration choice close to uniform conditional on
the migration cost — the household chain stays well-behaved under
mass-redistribution from migration. Lowering ε makes the chain more
responsive to small price gradients (economically richer,
computationally harder).

**Asymmetric calibrations need extra care.** A non-zero productivity
gradient produces strong migration-driven oscillations under damped
Picard: when one location has slightly higher prices, mass flows
there, depressing prices; mass then flows back. Period-3 orbits
appeared in early experiments at `A_home / A_abroad = 1.05 / 0.95`
even with `damping = 0.1`. Anderson acceleration or Newton on the
residual would converge in those regimes; this example stays in the
symmetric regime for simplicity.

## Result

```
K_home    = K_abroad     = 2.8305
K_total                  = 5.6610
pop_home  = pop_abroad   = 0.5000
r_home / r_abroad         = 3.87 %
w_home / w_abroad         = 1.1946
```

in 3 outer iterations (`damping = 0.1`, `tol = 0.25`).

The total capital `K_home + K_abroad = 5.66` is slightly below the
single-location Aiyagari `K = 5.69` at this calibration. Each
location has `L_each = L / 2 = 0.5`; symmetric productivity means
each market clears like a half-population Aiyagari, but
migration-induced mass mixing each period keeps the ergodic wealth
distribution slightly less concentrated near the borrowing
constraint than the single-location problem.

## Run

```bash
julia --project=. examples/spatial/steady_state.jl
```

About 10 seconds at `N_w = 400` (3 outer iterations × per-iteration
chain solve on a 400 × 3 × 2 state space).

## Files

- `model.jl` — params, layout (3 axes), the four-stage chain, prices.
- `steady_state.jl` — outer damped Picard on `(K_home, K_abroad)`.
- `../notebooks/spatial.jl` — Pluto-style four-section walkthrough.
