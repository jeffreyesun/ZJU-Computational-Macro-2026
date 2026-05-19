# HouseholdStages

A Julia package for the **household layer** of heterogeneous-agent
macro models: small, allocation-free *stages* you compose into
within-period chains, with forward / reverse-mode AD and
sequence-space Jacobian utilities built into the same interface.

The within-period problem of a heterogeneous-agent household model
decomposes cleanly into a sequence of operations: a Markov shock here,
a wealth update there, a discrete choice over locations, a continuous
consumption-savings argmax. `HouseholdStages` turns that
decomposition into the unit of code. Each stage carries its own
`backward!` (push `V` through the adjoint kernel `Kᵀ`) and `forward!`
(push `Λ` through the kernel `K`), composes under `∘ₛ`, and lifts to
moments (`lift_moments`), AD (`lift_jacobian`), and GPU (`lift_gpu`)
without leaving the stage interface.

The package is the implementation companion to the spin-off paper
`stages_paper/`, which writes down the categorical theory (stages as
morphisms in a category of state spaces, composition associative,
per-stage lifts functorial, V/Λ duality from adjointness, the
fake-news algorithm = forward-mode Jacobian functor). The library is
operational; the paper is the theory.

## Install

While the package lives inside this monorepo (no standalone release
yet, not currently registered in General), develop it from its own
`Project.toml`:

```bash
$ julia --project=HouseholdStages
```
```julia
julia> using HouseholdStages
```

Or, equivalently, activate the workspace project at the repo root
(which dev-depends on `HouseholdStages/`):

```bash
$ julia --project=.            # at the repo root
```
```julia
julia> using HouseholdStages
```

A future standalone release will be:

```julia
julia> ]add HouseholdStages   # placeholder for future
```

Dependencies: `ForwardDiff` (for `lift_jacobian` forward-mode
rebuilds) plus stdlib `LinearAlgebra`. Julia 1.10+.

## Worked example: Aiyagari steady state

A three-stage chain — Markov income, deterministic wealth receipt,
consumption-savings argmax — assembled, lifted with a `K_supplied`
moment, and solved by tatonnement on aggregate `K`:

```julia
using HouseholdStages

@kwdef struct AiyagariParams
    β = 0.96; σ = 1.5; α = 0.36; δ = 0.08; L = 1.0
    y_grid = [0.6, 1.0, 1.4]
    P_y    = [0.7 0.2 0.1; 0.2 0.6 0.2; 0.1 0.2 0.7]
    N_w    = 200; w_min = 0.0; w_max = 100.0
end
Base.Broadcast.broadcastable(p::AiyagariParams) = Ref(p)

function aiyagari_main()
    p      = AiyagariParams()
    wgrid  = [exp(t) - 1 for t in range(0.0, log(p.w_max + 1); length = p.N_w)]
    layout = StateLayout(StateAxis(:wealth, continuous_grid(wgrid)),
                         StateAxis(:income, discrete_finite(p.y_grid)))

    u(c)            = c < 0 ? -Inf : (c^(1 - p.σ)) / (1 - p.σ)
    wealth_post(cell; env) = (1 + env[].r) * cell.wealth + env[].w * cell.income

    shock   = MarkovStage(layout; axis = :income, transition = p.P_y)
    receipt = WealthChangeStage(layout; wealth_post = wealth_post,
                           wealth_axis = :wealth, closure_deps = (:r, :w))
    savings = ConsumptionSavingsStage(layout; β = p.β,
                                 utility = (cell, c; env) -> u(c),
                                 wealth_axis = :wealth,
                                 monotone_search = :divide_conquer)

    hh = lift_moments(shock ∘ₛ receipt ∘ₛ savings;
                      K_supplied = at_end(integrand = :wealth, reduce = sum))

    buffers = allocate(hh)
    dims = layout_size(layout)
    V    = zeros(dims...);   Λ = fill(1 / prod(dims), dims...)
    K    = 5.0
    for _ in 1:200
        r = p.α * (K / p.L)^(p.α - 1) - p.δ
        w = (1 - p.α) * (K / p.L)^p.α
        env = (; K, r, w)
        for _ in 1:1500
            V_new = backward!(hh, V, env, buffers)
            maximum(abs, V_new .- V) < 1e-7 && (V .= V_new; break)
            V .= V_new
        end
        for _ in 1:20_000
            Λ_new = forward!(hh, Λ, buffers)
            maximum(abs, Λ_new .- Λ) < 1e-6 && (Λ .= Λ_new; break)
            Λ .= Λ_new
        end
        Ksup = compute_moments(hh, env).K_supplied
        abs((Ksup - K) / K) < 0.02 && break
        K += 0.05 * (Ksup - K)
    end
    return K
end

@show aiyagari_main()        # K ≈ 5.66
```

(Wrapping the loop in a function makes the script work as a copy-pasted
file as well as in the REPL — Julia's soft-scope rule otherwise rejects
the rebinding of `K` inside the outer `for`.)

A polished version (exponential wealth grid, tatonnement scaffolding)
lives at `HouseholdStages/examples/aiyagari/` and converges to
`K = 5.6852` at `N_w = 400`.

## The stage catalog

Primitive stages, by category:

- **Exogenous transitions.** `MarkovStage` (Markov draw along a named
  axis; the canonical V/θ-independent stage).
- **Discrete choice.** `ArgmaxStage` (hard categorical choice over a named
  axis; kernel = integer policy). `LogitChoiceStage` (Gumbel-smoothed
  choice; kernel = probability tensor). `MigrationStage` (dedicated
  cost-matrix logit on a location axis — no user closure for the
  cost, just a matrix + dispersion ε).
- **Wealth dynamics.** `WealthChangeStage` (deterministic wealth transition
  via a `wealth_post(cell; env)` closure; backward linear-V
  interpolation, forward share-based redistribution).
  `AssetPriceChangeStage` (sugar over `WealthChangeStage` for `b_post = b_pre +
  (q − q_last) · h`). `ConsumptionSavingsStage` (choice of `b_end` on the
  wealth grid; implicit budget `c = b_in − b_end`; monotone-policy
  argmax with `:sequential` or `:divide_conquer` inner walk).
  `BorrowingConstraintStage` (`−∞` mask on infeasible cells; identity on
  Λ; supports both precomputed boolean masks and env-dependent
  closures).
- **Glue.** `UtilityStage` (additive state-only flow utility;
  identity on Λ; also serves as the *terminal* / bequest stage with
  `V_end = 0`). `IdentityStage` (no-op). `ForgetfulSumStage` (drop an
  axis; the canonical layout-changing stage).

**Composition.** `chain = a ∘ₛ b ∘ₛ c` produces a `ChainStage`
(allocation-free `@generated` unroll). **Product.** `prod = a ×ₛ b`
forms a Cartesian product over independent state axes, with kernel
the direct sum. **Replication.** `replicate_age(stage, N)` lifts a
stage to a life cycle by stacking it along an `:age` axis.

**Lifts.**

- `lift_moments(chain; spec_name = at_end(integrand, reduce, …))`
  attaches moment-emission closures to the chain; read them off
  after one forward pass via `compute_moments(chain, env)`.
- `lift_jacobian(stage; mode = :forward, n_dual, …)` rebuilds a stage
  with `ForwardDiff.Dual`-typed buffers for forward-mode
  differentiation. Per-stage `backward_adjoint!` / `forward_adjoint!`
  exist for reverse-mode (envelope theorem at the materialized `K`).
- `lift_gpu(stage)` rebuilds the stage with `cu(field)` on
  array-typed static fields. The methods exist; no GPU is wired in
  the workspace environment.

**Sequence-space utilities.** `expectation_vectors(chain, integrand,
T, buffers)` realizes Step 2 of the SSJ fake-news
algorithm (iterate `forward_adjoint!` to propagate an integrand
under `Kᵀ`). `build_F(curlyY, curlyD, curlyE)` is Step 3 (assemble
the fake-news matrix). `J_from_F(F)` is Step 4 (anti-diagonal
cumulation into the sequence-space Jacobian). Step 1 — per-period
direct effects — is model-specific and lives in the example.

## Worked examples (this repo)

Four self-contained examples under `HouseholdStages/examples/`. Each owns its
outer loop end to end; the library supplies stages and lifts only.

| Example | Chain | Solver | Headline |
|---|---|---|---|
| `aiyagari/` | `MarkovStage ∘ₛ WealthChangeStage ∘ₛ ConsumptionSavingsStage` | tatonnement on K | `K = 5.6852` |
| `krusell_smith/` | same chain, K-S parameters (employed/unemployed `y_grid`, larger `w_max`, K-S `δ`) | tatonnement on K | `K = 12.88` |
| `aiyagari_mit_shock/` | same chain, plus a transition path | damped Picard on `{K_t}` + SSJ T×T fake-news Jacobian | `K[5] ≈ 5.86` peak |
| `spatial/` | adds `MigrationStage` between income shock and receipt | damped Picard on `(K_home, K_abroad)` | `K_home = K_abroad = 2.8305` |

All four converge end-to-end at `N_w = 400`; the test suite is at
`450 / 450`.

## Where to read more

- **`STAGES_ARCHITECTURE.md`** (this folder) — design notes:
  struct discipline, parametric typing, allocation conventions,
  `Param{T}` and `env` slicing, the V/Λ duality identity as the
  free correctness test, the categorical framing where
  load-bearing.
- **`stages_paper/`** (one level up) — the spin-off paper folder.
  Two drafts in flight at the time of writing
  (`categorical_frarming/draft.tex`, `new_framing/draft.tex`);
  the folder's own files name the active version. Stages as
  morphisms in a category of state spaces; composition
  associative; lifts functorial; product Cartesian; duality from
  adjointness; the fake-news algorithm = forward-mode Jacobian
  functor `F_J`.
- **`HouseholdStages/examples/`** — four worked examples; the Aiyagari one
  is the smallest entry point.
- **`bench/runbenchmarks.jl`** — per-stage and chain benchmarks
  against hand-coded reference kernels; results in
  `bench/results.md`.

## Comparison to existing toolkits

`HouseholdStages` is closest in spirit to SSJ (Auclert, Bardóczy,
Rognlie, Straub 2021), whose `HetBlock` family exposes the same
household-layer decomposition; its stage primitives
(`Continuous1D / 2D`, `Exogenous`, `LogitChoiceStage`) instantiate the
same kernel-producing signature, the V/Λ duality identity is
implicit in the back / forward-pass structure, and the fake-news
algorithm is the same machinery as `expectation_vectors` +
`build_F` + `J_from_F`. The difference here is that composition
under `∘ₛ` and the per-stage functorial lifts (`lift_moments`,
`lift_jacobian`, `lift_gpu`) are first-class operations on stages
— chains compose by Julia operator, moments and Jacobians arise
as lifts of the chain, and the same interface carries forward
through the AD and GPU paths.

HARK (Carroll, Palmer, White et al.) decomposes period solvers
through class inheritance and `solveOnePeriod` chains, but stages
are implicit in the solver architecture, not the unit of code.
Reiter's perturbation framework operates on the linearized
backward operator and is conceptually stage-shaped (steady state
first, then perturb); the operator is constructed by hand, not
composed. Maliar–Maliar–Winant's intratemporal /
intertemporal decomposition is the same algebraic split,
expressed in a non-stage-shaped (iteration-on-allocation)
solver.

## License

MIT — see `LICENSE` at the repo root. *(2026-05-17: license
choice is a Claude default during overnight release prep; subject
to confirmation by the author before any public posting.)*
