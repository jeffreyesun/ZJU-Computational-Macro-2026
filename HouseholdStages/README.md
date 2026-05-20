# HouseholdStages

A Julia library for the **household layer** of heterogeneous-agent macro
models. The within-period problem decomposes into named *stages* — a
Markov shock, a deterministic wealth update, a discrete or continuous
choice — that compose under `∘`, lift to AD and GPU, attach named
moments, and plug into the sequence-space-Jacobian (SSJ) machinery
through one interface.

Every stage exposes the same **K-operator** signature: `backward!`
pushes a value function `V` through the adjoint `Kᵀ`, `forward!`
pushes a distribution `Λ` through `K`. Adjointness of `K` and `Kᵀ`
gives the duality identity

```
⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩
```

(where `r` is the stage's flow payoff). That identity is the
correctness test every stage gets for free.

The library is the operational companion to a separate
categorical-foundations paper. The paper proves the structural facts
the library uses — composition is associative, the per-stage lifts are
functorial, the fake-news algorithm is a forward-mode Jacobian functor —
but the library stands on its own.

## Install

```julia
julia> ]activate path/to/HouseholdStages
julia> ]instantiate
julia> using HouseholdStages
```

Dependencies: `ForwardDiff` (for `lift_jacobian` in forward mode) plus
stdlib `LinearAlgebra` and `Printf`. Julia 1.10 or later.

## Worked example — Aiyagari steady state

A three-stage chain (income shock, wealth receipt, consumption-savings
argmax) with the moment `K_supplied = ∫ wealth dΛ` attached, solved by
damped tatonnement on aggregate capital `K`.

```julia
using HouseholdStages

@kwdef struct AiyagariParams
    β :: Float64 = 0.96; σ :: Float64 = 1.5
    α :: Float64 = 0.36; δ :: Float64 = 0.08; L :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w :: Int = 400; w_max :: Float64 = 100.0
end
Base.Broadcast.broadcastable(p::AiyagariParams) = Ref(p)

u_crra(c, σ) = c < 0 ? -Inf : (σ == 1 ? log(c) : (c^(1 - σ)) / (1 - σ))

function aiyagari_household(p = AiyagariParams())
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid(0.0, p.w_max;
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
        utility         = (cell, c; env) -> u_crra(c, p.σ),
        wealth_axis     = :wealth,
        monotone_search = :divide_conquer,
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end

function aiyagari_prices(K, p = AiyagariParams())
    r = p.α * (K / p.L)^(p.α - 1) - p.δ
    w = (1 - p.α) * (K / p.L)^p.α
    return (; r, w)
end

function aiyagari_steady_state(; K_init = 5.0, update_speed = 0.05,
                                 rtol = 2e-2, max_iter = 500)
    p  = AiyagariParams()
    hh = aiyagari_household(p)
    K, V, Λ, err = K_init, nothing, nothing, Inf
    for iter in 1:max_iter
        env = (; K, aiyagari_prices(K, p)...)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        V, Λ = res.V, res.Λ
        K_sup = res.moments.K_supplied
        err   = abs(K_sup - K) / K
        err <= rtol && break
        K += update_speed * (K_sup - K)
    end
    return (; K, aiyagari_prices(K)..., V, Λ, err)
end
```

`aiyagari_steady_state()` returns `K = 5.685`, `r = 3.84%`, `w = 1.20`
in 18 outer iterations.

The example folder `examples/aiyagari/` ships a polished version with
verbose printing and CLI driver; `examples/notebooks/aiyagari.jl` is a
Pluto-style walkthrough.

## The stage catalog

A stage is a Julia struct whose K-operator is determined by its
configuration. Backward populates a per-call **kernel** (the K-operator's
runtime data — integer policy, choice-probability tensor, materialised
wealth-post array) and writes the new value function; forward consumes
the kernel and pushes the distribution. The library is organised
around the trichotomy `<X>StageSpec` (immutable configuration),
`<X>StageBuffer` (per-call state, including the kernel and a
`CacheState`), `<X>Stage` (the bundled user-facing wrapper). Only the
bundled `<X>Stage` form is exported.

### Exogenous transitions

- **`MarkovStage(layout; axis, transition)`** — Markov draw along a
  named axis. K is the transition matrix itself, V/θ-independent;
  kernel = `nothing`.

### Discrete choice

- **`ArgmaxStage(layout; choice_axis, flow_payoff, next_state_idx)`** —
  hard choice. K is a sparse permutation; kernel = integer policy.
- **`LogitChoiceStage(layout; choice_axis, flow_payoff, next_state_idx, ε)`** —
  Gumbel-smoothed choice. K is a stochastic kernel; kernel = action
  probability tensor.
- **`MigrationStage(layout; location_axis, migration_cost, amenity, ε)`** —
  dedicated cost-matrix logit on a location-style categorical axis.
  `migration_cost::Matrix` is `(n_loc, n_loc)`, `migration_cost[i, j]`
  is the cost of moving `i → j`. Optional `amenity` adds a
  destination-utility shifter: `nothing`, a static `Vector` of length
  `n_loc`, or a `(destination; env) -> Real` closure
  (env-dependent, materialised every backward pass).

### Wealth dynamics

- **`WealthChangeStage(layout; wealth_post, wealth_axis, extrap)`** —
  deterministic wealth update via a `wealth_post(cell; env)` closure.
  Backward interpolates `V_end` along the wealth axis at each cell's
  post-stage wealth; forward redistributes mass to the wealth grid via
  share-weighted accumulation. `extrap ∈ (:linear, :clip, -Inf)`
  controls the V-side extrapolation past the top of the grid;
  underflow / overflow on the Λ side always clip at the endpoints.
- **`AssetPriceChangeStage(layout; holdings_axis, ...)`** — sugar over
  `WealthChangeStage` for the asset-revaluation pattern `b_post =
  b_pre + (env.q − env.q_last) · cell.holdings_axis`. Returns a
  bundled `WealthChangeStage`; all `WealthChangeStage` machinery
  applies.
- **`ConsumptionSavingsStage(layout; β, utility, wealth_axis, monotone_search)`** —
  pick `b_end` on the wealth grid; implicit budget `c = b_in − b_end`.
  `monotone_search = :sequential` or `:divide_conquer`. The D&C path
  is `O(n_w log n_w)` per slice but assumes non-negative MPS (concave
  utility + linear budget gives this); the sequential path is the
  general fallback.
- **`BorrowingConstraintStage(layout; infeasible)`** — mask infeasible
  cells with `-Inf` on V; identity on Λ. `infeasible` is either a
  precomputed `AbstractArray{Bool}` of layout shape or a
  `(cell; env) -> Bool` closure (re-evaluated each backward pass).

### Glue

- **`UtilityStage(layout; utility)`** — additive flow utility on V;
  identity on Λ. K is the identity on measures. Also serves as a
  terminal / bequest stage: pass the bequest function as `utility`
  and seed the chain with `V_end = 0`.
- **`IdentityStage(layout)`** — no-op (K = I). Useful as a component
  inside `product` when one branch performs no within-period action.
- **`ForgetfulSumStage(layout; forget_axis)`** — drop one axis of the
  state space. Backward broadcasts V along the dropped axis; forward
  sums Λ along it. The canonical layout-changing stage.

## Composition, product, replication

```julia
chain = s1 ∘ s2 ∘ s3                                  # time-ordered
prod  = s1 × s2                                       # parallel
cohort_chain = replicate_age(chain, N; axis = :age)   # N uniform copies
```

- `∘` is `Base.:∘` overloaded on stages — **time-ordered**, opposite
  of Julia's function composition. `s1 ∘ s2` runs `s1` first.
- `×` builds a `ProductStage` along a new axis (default `:group`).
  The K-operator is the block-diagonal direct sum of the components';
  forward and backward operate per-component on slices of a fused
  tensor. v1 requires uniform component types and equal input layouts.
- `replicate_age(stage, N)` is sugar for `product(stage, stage, …,
  stage; axis = :age)`. Cross-cohort threading (bequest, birth,
  mortality) is the caller's responsibility.

`ChainStage` and `ProductStage` are themselves stages — closure under
`∘` and `×` makes compounds usable anywhere a primitive stage is.

## Moments

```julia
hh = chain
define_moments!(hh;
    K_supplied = at_end(integrand = :wealth,                       reduce = sum),
    L_supplied = at_end(integrand = (cell; env) -> cell.income,    reduce = sum),
)
m = compute_moments(hh, Λ, env)   # m.K_supplied, m.L_supplied
```

`define_moment!` / `define_moments!` are append-only by default; pass
`overwrite_existing_moment_definitions = true` to override.
`at_end(integrand, reduce; ...)`'s `integrand` is either a `Symbol`
(cell-field shortcut: `:wealth` ≡ `(cell; env) -> cell.wealth`) or a
closure following the standard `(cell, args...; env)` convention.
`reduce` is typically `sum` or `mean`.

`compute_moments(hh, Λ, env)` is non-mutating and takes `Λ` explicitly
— it doesn't read the chain's buffer.

## User-facing solvers

Three helpers absorb the per-env inner work (V backward to a fixed
point, Λ forward to its stationary distribution, plus moment readout)
so the consumer can focus on the outer loop:

- `solve_vfi_steady_state_given_env!(hh, env; V_init, tol, maxiter)`
  — repeated `backward!` to V's fixed point. Returns `(; V, iters,
  converged)`.
- `solve_lambda_steady_state_given_env!(hh; Λ_init, tol, maxiter)` —
  repeated `forward!` to Λ's stationary distribution. Kernels must
  have been seated by a prior `backward!` at the same env. Returns
  `(; Λ, iters, converged)`.
- `solve_steady_state_given_env!(hh, env; V_init, Λ_init, ...)` —
  bundles the above and also evaluates moments. Returns
  `(; V, Λ, moments, history, iters)`. The bundled chain warm-starts
  from buffer state across calls, so a sequence of perturbed-env
  solves runs in a few VFI iterations per call.

For transition paths:

- `solve_transition_given_env_path!(hh, env_path; Λ_0, V_T, max_inner_iters)` —
  allocates `T` per-period chain copies (sharing the Spec, each with
  its own Buffer), runs a backward sweep `t = T:-1:1` then a forward
  sweep `t = 1:T`, returns `(; V_path, Λ_path, moments_path, history,
  iters)`. The per-period buffer separation eliminates a class of
  stale-kernel bugs that hand-rolled transition drivers tend to hit.

For diagnostics:

- `compute_direct_jacobian!(hh, env_ss, T; inputs, outputs, eps)` —
  period-0 direct-effect finite differences of moments with respect
  to env, written on the diagonal of a `T × T` matrix. Diagonal-only:
  off-diagonal entries are zero by construction. **Treat as a
  calibration-time sanity check, not a sequence-space Jacobian.** The
  real fake-news Jacobian goes through the sequence-space utilities
  below.

Closing the model — tatonnement on `K` or `r`, AR(1) shock processes,
Anderson acceleration, calibration outer loops — is the consumer's
job. The library handles the household chain at a given env.

## Sequence-space utilities (fake-news algorithm)

Steps 2–4 of the SSJ fake-news algorithm
(Auclert-Bardóczy-Rognlie-Straub 2021):

- `expectation_vectors(hh, integrand, T)` — Step 2. Iterates the
  chain's `forward_adjoint!` (K-transpose action on a per-cell
  integrand) to produce expectation arrays for `t = 0, 1, …, T − 1`.
  The chain's kernels must have been seated by a prior `backward!`
  at the steady-state env; no env argument here.
- `build_F(curlyY, curlyD, curlyE)` — Step 3. Assembles the fake-news
  matrix from per-period direct effects (`curlyY`, `curlyD`, from
  Step 1) and the expectation vectors.
- `J_from_F(F)` — Step 4. Anti-diagonal cumulation into the
  sequence-space Jacobian.

Step 1 (per-period direct effects of a shock) is model-specific —
typically a finite-difference perturbation of env around the steady
state. `examples/aiyagari_mit_shock/ssj.jl` runs the full pipeline
end-to-end on the Aiyagari chain.

## Lifts

- `lift_jacobian(stage; mode = :forward, n_dual, tag, primal_eltype)` —
  rebuild with `ForwardDiff.Dual`-typed buffers for forward-mode AD,
  or expose the per-stage reverse-mode adjoint surface (`mode =
  :reverse` returns the stage unchanged; use `backward_adjoint!` /
  `forward_adjoint!` directly). The forward-mode rebuild flows through
  `with_eltype`, which is keyed on each Spec type — every concrete
  stage in the library has a `with_eltype` method.
- `lift_gpu(stage)` — not yet implemented; see Status.

The per-stage reverse-mode adjoints exist for every choice stage
(`ArgmaxStage`, `LogitChoiceStage`, `MigrationStage`,
`ConsumptionSavingsStage`) via the envelope theorem at the materialised
K. `WealthChangeStage`'s `forward_adjoint!` is wired (which is what
`expectation_vectors` needs); the `backward_adjoint!` on
`WealthChangeStage` is currently a stub — see Status.

## Worked examples

Four self-contained examples under `examples/`. Each owns its outer
loop end-to-end; the library supplies stages, lifts, and the per-env
inner solvers.

| Example | Chain | Outer loop | Result |
|---|---|---|---|
| `aiyagari/` | `MarkovStage ∘ WealthChangeStage ∘ ConsumptionSavingsStage` | tatonnement on `K` | `K = 5.685`, 18 iters |
| `krusell_smith/` | same chain, K-S calibration | tatonnement on `K` | `K = 12.88`, 24 iters |
| `aiyagari_mit_shock/` | same chain, permanent TFP step | damped Picard on `{K_t}` (`transition.jl`); SSJ pipeline (`ssj.jl`) | `K_ss^pre = 5.685` → `K_ss^post = 6.135`, peak `K[20] ≈ 6.05`, 15 transition iters |
| `spatial/` | adds `MigrationStage` between income shock and receipt | damped Picard on `(K_home, K_abroad)` | `K_home = K_abroad = 2.83`, symmetric pop split, 3 iters |

All four run at `N_w = 400`. Tests: 538 / 538 passing.

`examples/notebooks/` contains Pluto-style four-section walkthroughs
for each model — closer to a tutorial than a CLI driver.

## Status

Things that work but have rough edges, or are scaffolded for future
work:

- **`lift_gpu` is scaffolded but not implemented.** The entry points
  raise. The path forward: add CUDA as an optional / extension dep,
  define per-stage Spec methods that rebuild with `cu(field)` on
  array-typed static fields and re-allocate buffers, let
  algorithm-divergent methods (sparse scatter, monotone search)
  dispatch on the buffer's concrete array type.
- **`WealthChangeStage.backward_adjoint!` is stubbed.** Only
  `forward_adjoint!` is wired up — which is what
  `expectation_vectors` needs. The backward adjoint would mirror the
  share-based gather as a scatter; add when reverse-mode gradients
  through V on this stage are needed.
- **`Param`-keyed swept mode** (`Param(:symbol)` reading an env field
  at evaluation time) is supported but not exercised by any shipping
  example. Useful for estimation outer loops and sensitivity sweeps;
  unproven on a real workload.
- **`ProductStage` requires uniform components** — same concrete type,
  same input layout. Mismatched components raise at construction.
  Heterogeneous-shape products (e.g., working vs. retired cohorts
  with different state spaces) would need a separate dispatch path.
- **`compute_direct_jacobian!` is direct-effect only** — period-0 FD
  on the diagonal of a `T × T` matrix, off-diagonals zero. The full
  fake-news Jacobian goes through `expectation_vectors + build_F +
  J_from_F`. The function name advertises the scope.

## Comparison to existing toolkits

Closest in spirit to **SSJ** (Auclert, Bardóczy, Rognlie, Straub 2021),
whose `HetBlock` family exposes the same household-layer
decomposition. SSJ's primitives (`Continuous1D / 2D`, `Exogenous`,
discrete-choice helpers) instantiate the same kernel-producing
signature, the V/Λ duality is implicit in the back / forward-pass
structure, and the fake-news algorithm is the same machinery as
`expectation_vectors + build_F + J_from_F`. The difference: in
`HouseholdStages`, composition under `∘` and the per-stage
functorial lifts (`define_moment!`, `lift_jacobian`, `lift_gpu`,
`replicate_age`) are first-class operations on stages. Chains compose
by Julia operator, moments and Jacobians arise as lifts of the chain,
and the same interface carries forward through AD and GPU.

**HARK** (Carroll, Palmer, White et al.) decomposes period solvers
through class inheritance and `solveOnePeriod` chains, but stages are
implicit in the solver architecture rather than the unit of code.
**Reiter's** perturbation framework operates on the linearised
backward operator and is conceptually stage-shaped (steady state first,
then perturb), but the operator is constructed by hand rather than
composed. **Maliar-Maliar-Winant's** intratemporal / intertemporal
decomposition expresses the same algebraic split in a non-stage-shaped
(iteration-on-allocation) solver.

## Where to read more

- **`STAGES_ARCHITECTURE.md`** — design notes: the Spec/Buffer/Stage
  trichotomy, the K-operator framing, V/Λ duality as the free
  correctness test, kernel caching, allocation conventions, the
  Param-and-env machinery, conventions for stage authors.
- **`examples/`** — four worked examples; the Aiyagari one is the
  smallest entry point. `examples/notebooks/` has Pluto-style
  walkthroughs.
- **`bench/runbenchmarks.jl`** — per-stage and chain benchmarks
  against hand-coded reference kernels; results in `bench/results.md`.
  At `N_w = 400` the chain backward+forward sits at ~1.1×–1.2× the
  reference.

## License

MIT.
