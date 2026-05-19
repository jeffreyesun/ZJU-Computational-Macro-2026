# Aiyagari MIT shock — deterministic perfect-foresight transition

Aiyagari household block under a one-time unanticipated TFP shock with
AR(1) decay back to steady state. Demonstrates two distinct uses of the
`HouseholdStages` package on the canonical L03 / L04 three-stage
decomposition `IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavingsStage`:

1. **`transition.jl`** — straight perfect-foresight transition by
   damped Picard on the path of aggregate capital `K_t`.
2. **`ssj.jl`** — the household-layer sequence-space-Jacobian
   pipeline: `expectation_vectors`, `build_F`, `J_from_F`.

## Model

Same chain as `../aiyagari/`. State space wealth × income
(80 × 3); CRRA utility (`σ = 1.5`); β = 0.96; exponential wealth grid
on [0, 100]. TFP `A_t` is an explicit argument to `aiyagari_prices`,
so the transition driver can sweep it period by period; the household
chain's env stays minimal — `(;K, r, w)` — exactly matching the
steady-state example. `A_t` follows AR(1) decay from `A_0 = 1.05`
back to `A_ss = 1.0` with `ρ = 0.85`.

## Transition (`transition.jl`)

```julia
include("transition.jl")
tr = mit_shock_transition(; T = 100, A_0 = 1.05, ρ = 0.85,
                            damping = 0.2)
```

Algorithm:

1. Solve the deterministic steady state via tatonnement on K.
2. Guess `K_t = K_ss` for all `t`.
3. Backward pass: `V_{T+1} = V_ss`; walk back computing `V_t` under
   the period-t env `(K_t, r_t, w_t)`.
4. Forward pass: `Λ_1 = Λ_ss`; walk forward, reading `K^supplied_t`
   from `compute_moments` each period.
5. Damped update: `K_t ← (1 - d) K_t + d K^supplied_t`.
6. Iterate until `‖K^supplied - K‖_∞ < tol`.

Newton-on-path via the sequence-space Jacobian would converge much
faster than damped Picard, and is the natural extension of this
example using the SSJ utilities below.

### Expected result

Converges in ~20 outer iterations at `damping = 0.2` (the default,
post-`N_w = 400`). `K_path[1] ≈ 5.73` (impact), peaking around
`K[5] ≈ 5.86`, decaying back to `K_ss ≈ 5.69` by the end of the
horizon. Larger damping
(≥ 0.3) oscillates near the impact period and fails to converge — the
new chain's `WealthChangeStage ∘ₛ ConsumptionSavingsStage` diffuses mass more
slowly through `convert_distribution!`'s share-based push than the
old single-stage `GridSavings`, so the K-supplied response lags the
K-update more, and heavier damping is needed for stability.

### Performance

Roughly 2 seconds for the full transition (T = 100, including warm-
start steady-state solve), with ~30 backward and forward sweeps over
the chain per outer iteration. Compiling adds ~1s on first invocation.

## Sequence-space utilities (`ssj.jl`)

```julia
include("ssj.jl")
res = ssj_demo(T_horizon = 30)
```

Exercises the three household-layer SSJ utilities on the new 3-stage
chain:

* `expectation_vectors(chain, integrand, T, kernels, scratches)`
  — iterates `forward_adjoint!` on the chain to produce K-transpose-
  propagated expectation vectors over `T` periods. The `kernels`
  argument is populated by a prior `backward!` call at the steady
  state; no env is needed here because the K used in the K-transpose
  iteration is materialized at the SS eval point. Step 2 of the SSJ
  fake-news algorithm.
* `build_F(curlyY, curlyD, curlyE)` — assembles the fake-news
  matrix `F` from per-period direct effects (`curlyY`, `curlyD`) and
  expectation vectors (`curlyE`). Step 3.
* `J_from_F(F)` — cumulates along anti-diagonals into the sequence-
  space Jacobian `J`. Step 4.

### Library prerequisites

The new chain uses `WealthChangeStage` and `ConsumptionSavingsStage` in place
of the old `GridSavings`. `expectation_vectors` iterates the chain's
`forward_adjoint!` — so both stages must implement that adjoint. They
do: `ConsumptionSavingsStage.forward_adjoint!` is a sparse gather along
the policy (identical in structure to `GridSavings`);
`WealthChangeStage.forward_adjoint!` is a share-weighted gather (the dual
of `convert_distribution!`'s share-based scatter), implemented in
`HouseholdStages/src/lifts/jacobian.jl` as part of the 2026-05-15
refactor.

### Caveat — discrete-policy stages and finite-difference cross-checks

A natural validation is to compare `J[t, s]` from the SSJ pipeline
against finite-difference perturbation of the household chain at a
known steady state. **For an Aiyagari chain whose savings stage is
hard-argmax `ConsumptionSavingsStage`, a small-`ε` finite difference
produces zero**: the integer policy is locally invariant to sub-grid
price perturbations. Per-stage reverse-mode adjoints in
`lift_jacobian.jl` handle this via the envelope theorem (subgradient
at boundary cells); the SSJ pipeline inherits the same property.

A meaningful FD cross-check requires a smoothed choice stage —
`LogitChoiceStage` with a non-trivial `ε`. That's a natural variant to
write next; this example does not include it.

## Files

| File | Purpose |
|---|---|
| `model.jl` | Stages + TFP path + Cobb-Douglas prices `aiyagari_prices(K, A, p)` |
| `transition.jl` | Steady-state warm start + perfect-foresight transition (damped Picard) |
| `ssj.jl` | Sequence-space-Jacobian utilities demo |
| `../notebooks/aiyagari_mit_shock.jl` | Notebook-style four-section walkthrough |
