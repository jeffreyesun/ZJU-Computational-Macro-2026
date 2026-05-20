# Aiyagari MIT shock — transition path + sequence-space utilities

The Aiyagari household block under a one-time unanticipated permanent
TFP step. Two distinct uses of the same chain:

- **`transition.jl`** — solves the deterministic perfect-foresight
  transition by damped Picard on the path of aggregate capital
  `{K_t}` using `solve_transition_given_env_path!`.
- **`ssj.jl`** — exercises the household-layer sequence-space-Jacobian
  utilities `expectation_vectors`, `build_F`, `J_from_F` at the
  steady state.

The chain is the same three-stage decomposition as `../aiyagari/`:

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

TFP `A_t` is carried as an explicit argument to `aiyagari_prices` (not
baked into the chain) so the transition driver can sweep it period by
period; the household chain's env stays minimal — `(; K, r, w)`.

## The shock

A **permanent step**: `A_t = A_0` for all `t ≥ 1` with default
`A_0 = 1.05`. The economy starts in the deterministic steady state at
`A = 1.0` and transitions to a new steady state at the higher TFP
level. (`model.jl`'s `tfp_path` returns the constant path; AR(1)
mean-reversion would be a one-line edit.)

## Transition algorithm (`transition.jl`)

```julia
tr = mit_shock_transition(; T = 100, A_0 = 1.05,
                            damping = 0.2, tol = 1e-3)
```

1. Solve the pre-shock steady state at `A = 1`. Pins `K_ss^pre`,
   `V_ss^pre`, `Λ_ss^pre`.
2. Solve the post-shock steady state at `A = A_0`. Pins `K_ss^post`,
   `V_ss^post`.
3. Initialise `{K_t}` as a linear interpolation `K_ss^pre →
   K_ss^post`.
4. Iterate the damped Picard step:
   - Build `env_path[t] = (; K = K_t, aiyagari_prices(K_t, A_t)...)`.
   - One call to `solve_transition_given_env_path!(hh, env_path;
     Λ_0 = Λ_ss^pre, V_T = V_ss^post)` runs the per-period backward
     sweep then forward sweep with per-period buffers, and returns
     `tr.moments_path[t].K_supplied` for each `t`.
   - Damped update: `K_t ← (1 − d) K_t + d K_supplied_t`.
5. Stop when `‖K_supplied − K‖∞ < tol`.

The per-period buffer separation inside
`solve_transition_given_env_path!` is what makes step 4 safe:
each forward pass reads the kernel materialised by *its own* backward
pass at the period-specific env, not the stale kernel from a
neighbouring period.

## Transition result

```
K_ss^pre     = 5.6847
K_ss^post    = 6.1352
K[1]   (impact)  = 5.7111
K[5]             = 5.8348
K[20]            = 6.0538
K[50]            = 6.1284
K[100] (≈end)    = 6.1340
```

Converges in 15 outer iterations at `damping = 0.2`, `tol = 1e-3`.
Larger damping (≥ 0.3) oscillates near the impact period — the new
three-stage chain diffuses mass more slowly through share-based
redistribution than a single-stage `GridSavings` would, so the
K-supplied response lags the K-update more and needs heavier damping
for stability.

## Sequence-space utilities (`ssj.jl`)

Steps 2–4 of the SSJ fake-news algorithm
(Auclert-Bardóczy-Rognlie-Straub 2021) on the post-refactor chain:

```julia
ssj_demo(; T_horizon = 30)
```

1. Solve the pre-shock steady state.
2. Re-seed the chain's kernels at the steady-state env via one
   `backward!` / `forward!` pair (so the K used in the K-transpose
   iteration is the SS K).
3. **Step 2.** `𝓔 = expectation_vectors(hh, cell -> cell.wealth,
   T_horizon)` iterates the chain's `forward_adjoint!` to produce
   the K-transpose-propagated expectation arrays.
4. **Step 3.** Synthesise per-period direct effects (`curlyY`,
   `curlyD`) — the script uses a placeholder pattern; a real
   application populates these by finite-difference perturbation of
   env around the steady state. Then `F = build_F(curlyY, curlyD,
   curlyE)`.
5. **Step 4.** `J = J_from_F(F)` cumulates along anti-diagonals.

The SS-correctness check `⟨𝓔[t], Λ_ss⟩ ≈ K_ss` holds for all `t` (to
~4 decimals at `N_w = 400`) — `𝓔[t]` is the K-transpose-propagated
expectation of the wealth integrand `t` periods out, which against
the stationary distribution returns the same steady-state aggregate
the Λ-side `compute_moments` does.

### What `expectation_vectors` requires

`forward_adjoint!` methods on every component of the chain. The
library ships them for `MarkovStage`, `WealthChangeStage`,
`ConsumptionSavingsStage`, `LogitChoiceStage`, `MigrationStage`,
`ArgmaxStage`, `IdentityStage`, `ForgetfulSumStage`, and the
`ChainStage` / `ProductStage` walks. `WealthChangeStage`'s
`backward_adjoint!` is currently a stub — not needed by the
sequence-space pipeline (which only uses `forward_adjoint!`); see
the package-level Status section.

## Discrete-policy stages and finite-difference cross-checks

A natural validation is to compare `J[t, s]` from the SSJ pipeline
against a finite-difference perturbation of the household chain at
the steady state. For an Aiyagari chain whose savings stage is
hard-argmax `ConsumptionSavingsStage`, a small-ε FD produces zero —
the integer policy is locally invariant to sub-grid price
perturbations. The per-stage adjoints in `lift_jacobian.jl` handle
this via the envelope theorem (subgradient at boundary cells); the
SSJ pipeline inherits the same behaviour. A meaningful FD
cross-check requires a smoothed choice stage —
`LogitChoiceStage`-based savings with non-trivial `ε`. That's a
natural next variant; this example does not include it.

## Run

```bash
julia --project=. examples/aiyagari_mit_shock/transition.jl
julia --project=. examples/aiyagari_mit_shock/ssj.jl
```

The transition takes ~30 s at `T = 100`, `N_w = 400`; the SSJ demo
takes a few seconds after the same warm-up.

## Files

- `model.jl` — params, layout, stage constructors,
  `aiyagari_prices(K, A, p)`, `tfp_path`.
- `transition.jl` — pre- and post-shock SS warm-starts plus the
  damped Picard transition loop.
- `ssj.jl` — sequence-space-Jacobian utilities demo.
- `../notebooks/aiyagari_mit_shock.jl` — Pluto-style four-section
  walkthrough.
