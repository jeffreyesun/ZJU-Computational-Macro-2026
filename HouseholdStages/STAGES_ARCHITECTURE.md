# Stage architecture — design notes

This is the design doc for `HouseholdStages`. It covers struct
discipline, method signatures, allocation, dispatch, and conventions
— the operational counterpart to the categorical theory in the
companion paper.

`HouseholdStages` is a Julia library demonstrating efficient
computation of heterogeneous-agent macro models: composable
**stages** assembled into a within-period household block, the
**K-operator** framing they share, V/Λ envelope-duality, per-stage lifts
(`lift_jacobian`, `lift_gpu`, `replicate_age`), moment attachment
(`lift_moments`), and the sequence-space utilities for
fake-news-algorithm Jacobians.

## Document organization

1. Stages, the K-operator, and composition (abstract picture).
2. The Julia realization — structs, methods, allocation, parametric
   typing.
3. Patterns the API enables (`Param`, closures, `cells`,
   `axis_position`, `env` slices, default conventions).
4. Design requirements that shaped the library.
5. User-facing API — Aiyagari example, end-to-end self-contained.
6. Per-stage lifts and moment attachment — operational sketches.
7. Sequence-space utilities.
8. Open questions.
9. Non-goals.

---

## 1. Stages, the K-operator, and composition

A **stage** is a configuration struct that, given a continuation
value `V_out` and an `env` slice (prices, calibration), computes
a backward pass producing `V_in = r + Kᵀ V_out` and a forward pass
producing `Λ_out = K Λ_in`, where `K ∈ Hom(M(S_in), M(S_out))` is
the stage's **kernel** — a linear operator on measures, produced
at the evaluation point. The kernel is the data the stage's
`kernel` slot stores; it's what backward and forward both consume
(the same operator, via direct vs. adjoint action). **Naming
convention:** `kernel` always refers to the K data object (the
output); the stage's `backward!` method is the function that
produces it from `(V_out, env)`.

Stages compose under `∘ₛ` (sequential composition in time) and
product under `×ₛ` (parallel composition across components — cohorts,
types, sectors). `Stage` is closed under both, so any compound built
from primitives is itself a stage. Per-stage lifts (`lift_jacobian`,
`lift_gpu`, `replicate_age`) wrap stages individually; the chain's
machinery inherits the wrap automatically. Moment attachment via
`lift_moments` is a separate concept — see §6.

### 1.1 State spaces, V/Λ duality

A stage acts on a **state space** `S` (a tensor of grid points, each
axis named and typed — see §1.5). Two function spaces live over `S`:

- `V(S)` — real-valued functions on `S`. A value function.
- `M(S)` — (signed) measures on `S`. A distribution.

`V(S)` and `M(S)` are dual under `⟨V, Λ⟩ = ∫_S V(s) dΛ(s)`. Aggregates
("moments") are linear functionals `M(S) → ℝ`.

A stage's input and output state spaces may differ: a stage can drop
or add axes. `ForgetfulSum` is the worked example — its backward
broadcasts along the dropped axis, its forward sums along it, and the
duality identity holds.

### 1.2 The K-operator

A stage `σ` produces, at the evaluation of `(V_out, θ)`:

```
K  =  σ.get_kernel(V_out, θ)  ∈  Hom(M(S_in), M(S_out))   (linear operator on measures)
r  =  σ.get_payoff(V_out, θ)  ∈  V(S_in)                  (flow payoff over the input)
```

**`K` is what the `kernel` field stores.** Different stage classes
materialize `K` differently — the kernel field's content is whatever
small data parameterizes K, not K-as-a-dense-matrix:

| Stage class | What K is | Kernel content |
|---|---|---|
| `MarkovAlong` (pure Markov) | Transition matrix, V/θ-independent | `nothing` (matrix lives on the stage struct) |
| `Argmax` (hard discrete choice) | Sparse permutation `δ_{π(s)}` | Integer policy `π::Array{Int}` |
| `LogitChoice` (smoothed) | Stochastic kernel `Σ_a π(a\|s) δ_{σ(s,a)}` | Probability tensor `P::Array{Float64}` |
| `Migration` (logit on a location axis) | Stochastic kernel from a cost-matrix + ε softmax | Probability tensor `P::Array{Float64}` |
| `WealthChange` (deterministic wealth update) | Per-cell deterministic map; backward linear-V interp | `nothing` (interp is on-the-fly) |
| `ConsumptionSavings` (continuous savings argmax) | Sparse-stochastic via on-grid argmax | Integer next-wealth policy |
| `ForgetfulSum` (layout-changing) | Sum-along-axis operator | `nothing` (structural) |
| `IdentityStage` | Identity operator on M(S) | `nothing` (structural) |
| `UtilityStage` | Identity on M(S); flow payoff on V | `nothing` (utility evaluated inline) |
| `BorrowingConstraint` | Identity on M(S); `-Inf` mask on V | `nothing` (mask applied inline) |

Backward and forward both consume `K`, just via different actions:

```
V_in   =  r + Kᵀ V_end       (backward: adjoint action on functions)
Λ_end  =  K Λ_start          (forward: direct action on measures)
```

The duality identity `⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩`
follows from K-Kᵀ adjointness in the `L²(V) ⊗ L²(M)` pairing. Every
stage that defines a K-operator inherits this identity for free; the
correctness test is built in.

### 1.3 Composition (`∘ₛ`)

Sequential composition `σ_1 ∘ₛ σ_2` produces a stage whose K is the
operator composition `K_2 ∘ K_1`. Both stages see the same θ; the
chain only makes sense at a single coherent env. Composition is
associative; the chain doesn't materialize `K_1 ∘ K_2` explicitly —
it walks backward through stages applying `K_iᵀ` to accumulate
`V_in`, then forward applying `K_i` to push Λ.

`Stage` is closed under `∘ₛ`. The `StageChain` struct holds a tuple
of stages and implements the stage interface; a compound built from
primitives is itself a stage.

### 1.4 Product (`×ₛ`)

The Cartesian product `s_1 ×ₛ s_2 ×ₛ ... ×ₛ s_n` produces a stage on
the product state space `S_1 × S_2 × ... × S_n × axis` whose K is the
direct sum (block-diagonal) of per-component K's. Components are
independent: forward and backward run per-component in parallel; the
kernel is a tuple of per-component kernels; the duality identity
holds component-wise.

For uniform components (same struct type, same layout, only field
values differ) the library uses a **fused-tensor representation**:
one big buffer with the product axis as an outer dimension; per-
component buffers are *views* into the fused storage. Type-uniformity
dispatch picks this path automatically.

Heterogeneous-shape components (different layouts per slot) are not
supported. The constructor raises with a pointer to the relevant
substitute pattern; see §3.4.

### 1.5 Layouts

A `StateLayout` is an ordered tuple of `StateAxis`. Two axis kinds:

- **`continuous_grid`** — a grid of `Float64` points used for
  interpolation (`wealth`, `productivity`, fine-grained housing).
- **`discrete_finite`** — a finite vector of levels. Element type is
  any leaf type (`Float64`, `Int`, `Symbol`).
  - `discrete_finite([0.0, 1.0, 2.0])` for real-valued discrete
    levels (e.g., housing where `housing == 0.0` means renter).
  - `discrete_finite([:NYC, :LA, :Chicago])` for symbolic location
    labels.
  - `discrete_finite([1, 2, 3, 4, 5])` for sector indices.
  - `categorical(...)` is sugar for `discrete_finite([:sym1, ...])`.

Axis values are accessed via `layout.foo.grid` (continuous) or
`layout.foo.levels` (discrete). Library helper `axis_position(layout,
:foo)` returns the integer dimension number.

### 1.6 `env` and the per-stage slice

`env` (the household's "economic environment") is a `NamedTuple`
carrying everything that isn't V/Λ: prices, calibration scalars,
aggregate-state coordinates, AD perturbations, swept-parameter values.
`env` is shared across all stages in a household chain; it is the
*only* runtime configuration passed to `backward!`. In the math
sections this is the same object as `θ ∈ Θ_run` — the code variable
name is `env`, the math symbol is `θ`.

Each stage declares its **`env` slice** — the `env` fields the stage
reads. This serves three purposes:

1. **Static checking.** At chain construction, the union of per-stage
   slices is computed as the chain's `env_type`; the user-supplied
   env is checked for shape compatibility. Wrong-shaped env errors at
   construction, not at first call.
2. **Change-set propagation.** When the outer solver perturbs only a
   subset of `env` fields, only stages whose slice intersects the
   change are re-run. This is the SSJ HetBlock pattern.
3. **F_J slicing.** Differentiating with respect to an `env`
   coordinate only requires propagating tangents through stages whose
   slice contains that coordinate. Per-stage adjoint cost scales with
   declared deps, not chain length.

The `env` slice is computed at stage construction as the union of:

- `static_env_deps(::Type{S})` — fields the stage *type* itself
  reads, hardcoded once per stage class.
- `closure_deps` (kwarg passed at construction) — fields read by
  user-supplied closures (utility, budget, flow_payoff, moment
  integrand).
- Symbol-valued `Param` fields — see §2.3.

### 1.7 User closures: `f(cell, args...; env)`

User closures (utility, `wealth_post`, `flow_payoff`, moment
integrand) follow a single signature convention: `cell` is the first
positional argument; any stage-specific positional arguments
(consumption `c`, action index, savings choice) come next; `env` is
the only kwarg.

```julia
# Utility — `cell` first, consumption `c` next, `env` kwarg.
utility = (cell, c; env) -> (c^(1 - env.σ)) / (1 - env.σ)

# Deterministic wealth update — `cell` first; `env` is passed as a
# Ref by WealthChange's broadcast, so unwrap it with `env[]`.
wealth_post = (cell; env) -> (1 + env[].r) * cell.wealth + env[].w * cell.income

# LogitChoice flow payoff — `(cell, action; env)`.
flow_payoff = (cell, action; env) -> action == 0.0 ?
                                     -ϕ * env.q * cell.housing : 0.0
```

`cell` is a `NamedTuple` of axis values (`Float64` / `Int` / `Symbol`
leaves) over the current `StateLayout`. Each stage stores the user's
closure as a typed field (e.g., `flow_payoff :: F`) with the function
type as a type parameter; broadcasts treat closures as scalars by
default. There is no wrapper struct around the closure.

Declare which `env` fields the closure reads via the `closure_deps`
kwarg at construction:

```julia
WealthChange(layout;
    wealth_post  = (cell; env) -> (1 + env[].r) * cell.wealth + env[].w * cell.income,
    wealth_axis  = :wealth,
    closure_deps = (:r, :w))
```

This feeds the per-stage `env` slice (§1.6).

---

## 2. The Julia realization

This section is the operational counterpart to §1 — how the abstract
picture lands in code.

### 2.1 Stage struct discipline

Stages are concrete structs subtyping `AbstractStage` (a dispatch
tag, see §2.6). All array- and function-typed fields are **parametric
type variables**; no abstract field types anywhere in the public
stage interface.

```julia
struct MarkovAlong{M<:AbstractMatrix, T<:Real, N,
                   LIn<:StateLayout, LOut<:StateLayout,
                   AV<:AbstractArray{T,N}} <: AbstractStage
    transition    :: M
    axis          :: Symbol
    axis_dim      :: Int            # resolved at construction
    input_layout  :: LIn
    output_layout :: LOut
    V_start       :: AV
    Λ_end         :: AV
end

struct LogitChoice{F, BF,
                   LIn<:StateLayout, LOut<:StateLayout,
                   N, D, T<:Real, AV<:AbstractArray{T,N}} <: AbstractStage
    choice_axis    :: Symbol
    choice_dim     :: Int
    flow_payoff    :: F                   # user closure, stored directly
    next_state_idx :: BF                  # function (cell, action) -> next-cell index
    ε              :: Param{T}            # see §2.3
    closure_deps   :: NTuple{D, Symbol}
    input_layout   :: LIn
    output_layout  :: LOut
    V_start        :: AV
    Λ_end          :: AV
end
```

The point of the parametric types is type-stable hot paths — every
field access resolves to a concrete type at compile time.

### 2.2 Method signatures

Every stage implements:

```julia
backward!(stage, V_end, env, kernel, scratch) -> V_start
forward!(stage, Λ_start, kernel, scratch, moments) -> Λ_end
allocate(stage, T) -> (kernel, scratch)
static_env_deps(::Type{Stage}) -> NamedTuple
```

`backward!` writes into the stage's pre-allocated `V_start` buffer
(accessed via `V_start_buffer(stage)`) and into `kernel` (which
records the evaluated K-operator data — the policy index, the choice
probability tensor, etc.). `forward!` reads `kernel`, writes into
`Λ_end`. **`forward!` does not take `env`** — env was fully consumed
by `backward!` in producing the kernel. Both methods return the
buffer they wrote to.

`allocate(stage, T)` produces `(kernel, scratch)` workspace:

- **`kernel`** holds the runtime data parameterizing the K-operator
  at the current `(V_out, env)`. Lifetime spans backward → forward;
  cannot be aliased across stages. For stages whose K is V/θ-
  independent and lives on the struct (`MarkovAlong.transition`,
  `IdentityStage`, `ForgetfulSum`, `UtilityStage`), `kernel` is
  `nothing`.
- **`scratch`** is anything else the stage needs internally —
  permuted views, intermediate sums, working arrays. Carries no
  morphism content; may be aliased across stages of matching shape.

The stage's `V_start` and `Λ_end` buffers are stage-allocated by
layout (not user-written); accessible via `V_start_buffer(stage)`
and `Λ_end_buffer(stage)`.

**Convenience forms.** None. Every `backward!` / `forward!` /
`backward_adjoint!` / `forward_adjoint!`
call must explicitly pass the `(kernel, scratch)` workspace. This
avoids hidden allocations and makes the kernel/scratch discipline
visible at every call site. Tests and examples allocate workspace
explicitly via `kernel, scratch = allocate(stage)` and pass it
through.

### 2.3 `Param{T}` for calibrated-or-swept parameters

```julia
struct Param{T}
    val :: Union{T, Symbol}
end

resolve(p::Param{T}, env) where T =
    p.val isa Symbol ? env[p.val] : p.val
```

`Param(0.25)` is a calibrated value. `Param(:ξ_housing)` reads from
`env.ξ_housing`. Mode-flip (calibrated ↔ swept) is a field
mutation: `stage.ε = Param(:ξ_housing)`. No stage rebuild, no chain
reconstruction. The dependency walker collects `Param`-typed fields
whose `.val isa Symbol`, unions with `closure_deps`.

Type stability: `Union{T, Symbol}` is a small union; Julia handles
it via union-splitting.

### 2.4 Cell iteration

```julia
for (idx, cell) in cells(layout)
    # idx :: NamedTuple of integer axis indices
    # cell :: NamedTuple of axis values (Float64 / Int / Symbol leaves)
    ...
end
```

Constructing the `NamedTuple` `cell` per iteration is stack-allocated
(bits-typed) and DCE-friendly for unused fields. The inner-loop
machine code is equivalent to hand-written index-based kernels with
proper `@inline` and `@inbounds`.

`cell_array(layout)` returns an N-D `Array{NamedTuple}` over the
layout — useful when broadcasting a closure over all cells. Power
users who want raw indices can use `CartesianIndices` directly;
`cells(layout)` and `cell_array(layout)` are the high-level helpers.

### 2.5 Stage composition and product as structs

```julia
struct StageChain{Stages<:Tuple} <: AbstractStage
    stages :: Stages
    # env_type computed at construction from per-stage env_slices
end

struct ProductStage{Stages<:Tuple} <: AbstractStage
    stages :: Stages
    axis   :: Symbol
end
```

Both subtype `AbstractStage` (§2.6) and implement `backward!`,
`forward!`, `allocate`. The chain walks its tuple in reverse for
backward and forward for forward (via a `@generated` unroll for zero
allocation); the product delegates per-component.

`StageChain` and `ProductStage` are construction representations; in
prose, both are *stages* (closure under `∘ₛ` and `×ₛ`).

### 2.6 `AbstractStage` is a tag, not a hierarchy

Some dispatch convenience benefits from a common supertype:

```julia
abstract type AbstractStage end
```

It carries *no* behavior beyond what's needed for dispatch on "this
is a stage." All methods are free functions specialized to concrete
stage types. The supertype exists for collections (`Tuple{Vararg{
AbstractStage}}`) and chain dispatch; nothing more.

### 2.7 CPU/GPU dispatch

Stages are parametric on their array-typed fields. A
`MarkovAlong{Matrix{Float64}, ...}` and a
`MarkovAlong{CuArray{Float64,2}, ...}` are different concrete
instantiations of the same struct definition.

```julia
lift_gpu(stage::MarkovAlong) = MarkovAlong(cu(stage.transition), ...)
lift_gpu(stage::LogitChoice) = LogitChoice(..., stage.flow_payoff, ...)
lift_gpu(chain::StageChain)  = StageChain(map(lift_gpu, chain.stages))
```

`allocate(stage, T)` produces workspace whose array types match the
stage's fields. Methods with platform-specific algorithms (e.g.,
sparse policy scatter on GPU vs. CPU) dispatch on the buffer's
concrete array type. One struct definition per stage concept; no
GPU sibling types.

### 2.8 Per-stage lifts

Library-provided per-stage lifts (operations that take a stage and
produce a new stage of the same kind, preserving composition):

- `lift_gpu(chain)` — rebuilds each stage with `cu(field)` on
  array-typed static fields. Per-stage methods exist as stubs that
  raise; full implementation deferred until GPU hardware is wired in.
- `replicate_age(chain, N_age)` — uniform replication via product:
  `product(chain, chain, ..., chain; axis = :age)` (N copies).
  Cross-cohort threading (bequest, birth, mortality) is the caller's
  responsibility, structurally identical to time fixed-point
  threading.
- `lift_jacobian(stage, mode)` — forward-mode realized by
  re-allocating buffers as `ForwardDiff.Dual`-typed; reverse-mode by
  per-stage adjoint methods (including the choice stages via the
  envelope theorem). Each preserves composition (the chain rule).

**Moment attachment** (`lift_moments`) is *not* a lift in this sense
— it doesn't produce another stage. It wraps a chain in a
`MomentedChain` carrier so that `compute_moments(chain, env)` can
read off named aggregates after a forward pass. See §6.

Theorem-grade statements (composition preservation for the actual
lifts) live in the companion paper; here, operational sketches and
contracts.

---

## 3. Patterns

### 3.1 Per-instance fields = same-type, different-parameter stages

Two `MarkovAlong` stages with different transition matrices are two
instances of the same struct type with different field values. Two
`LogitChoice` stages with different `ε` values likewise. The instance
is the namespace; the chain iterates generically.

### 3.2 `Param`-typed fields are sweepable without rebuild

A stage configured with `Param(0.25)` switches to swept mode with
one field mutation: `stage.ε = Param(:ξ_housing)`. The chain stays
identical; env now carries `:ξ_housing`. No realloc, no rebuild.
Pattern for estimation outer loops and sensitivity sweeps.

### 3.3 Moment specifications disambiguate by integrand

```julia
# Simple end-of-chain moment using a cell-field shortcut.
hh = lift_moments(chain;
    K_supplied = at_end(integrand = :wealth, reduce = sum),
)

# Per-location moment via a closure that reads cell coordinates.
hh = lift_moments(chain;
    K_home   = at_end(
        integrand = (cell; env) -> cell.location == :home ? cell.wealth : 0.0,
        reduce = sum),
    K_abroad = at_end(
        integrand = (cell; env) -> cell.location == :abroad ? cell.wealth : 0.0,
        reduce = sum),
)
```

Moment integrands follow the same `f(cell, args...; env)` convention
as flow_payoff / utility. `at_end` is the only moment-spec form
shipped; consumers split moments by writing integrand closures that
read cell coordinates.

### 3.4 Layout-changing stages declare both layouts

A stage's `input_layout` and `output_layout` need not match. The
chain's typing machinery validates that adjacent stages agree
(one's `output_layout == next's input_layout`) at construction.

`ForgetfulSum` is the canonical example — drops one axis. The dual
`Introduce` (adds an axis) is not currently shipped.

### 3.5 Default axis names

Library-provided stages default to:

| Default name | Typical kind | Used by |
|---|---|---|
| `:wealth` | `continuous_grid` | `WealthChange`, `ConsumptionSavings` |
| `:income` | `discrete_finite` (Float64) | `MarkovAlong` (income shock) |
| `:location` | `discrete_finite` (Symbol) | `Migration`, `Argmax`/`LogitChoice` |
| `:age` | `discrete_finite` (Int) | `replicate_age` product axis |
| `:group` | (varies) | Generic default for `product` |

Users with non-canonical layouts override via kwargs
(`wealth_axis = :k`).

---

## 4. Design features

1. **Standardized stage API.** Every stage implements the same
   `backward!` / `forward!` / `allocate` interface dispatched on
   concrete stage type. Generic composition is possible.
2. **Per-stage allocation.** `allocate(stage, T)` is per-stage; the
   chain assembles per-stage workspace. Adding a stage does not
   require editing a monolithic struct.
3. **Named axes resolved at construction.** `StateLayout` with named
   axes; `axis_position(layout, :name)` returns the integer at
   construction. No hard-coded `permutedims!` calls or hard-coded
   axis indices anywhere on the hot path.
4. **No global state.** Everything lives on the stage struct as
   parametric fields. No module-level transition matrices, grid
   sizes, or axis indices.
5. **Composition as a first-class operator.** `∘ₛ` and `×ₛ` produce
   stages; the chain walks the tuple of stages generically. Per-model
   hand-rolled `iterate_V_backward!` / `iterate_λ_forward!` are not
   necessary.

---

## 5. User-facing API — Aiyagari example

A complete Aiyagari steady state. The household chain comes from
`HouseholdStages`; the outer loop is plain Julia (the consumer
writes their own — see §7).

### 5.1 Household chain

```julia
using HouseholdStages

@kwdef struct AiyagariParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36
    δ :: Float64       = 0.08
    L :: Float64       = 1.0
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1; 0.2 0.6 0.2; 0.1 0.2 0.7]
    N_w   :: Int       = 400
    w_min :: Float64   = 0.0
    w_max :: Float64   = 100.0
end
Base.Broadcast.broadcastable(p::AiyagariParams) = Ref(p)

const params = AiyagariParams()

# Exponentially-spaced wealth grid: dense near zero (binding borrowing
# constraint) and coarse at the top. A uniform grid would amplify V
# through linear extrapolation past `w_max` each pass; the exp grid
# makes the top span wide enough that this doesn't bite.
exp_wealth_grid(lo, hi, n; shift = 1.0) =
    [exp(t) * shift - shift + lo
     for t in range(0.0, log((hi - lo + shift) / shift); length = n)]

layout = StateLayout(
    StateAxis(:wealth, continuous_grid(exp_wealth_grid(params.w_min, params.w_max, params.N_w))),
    StateAxis(:income, discrete_finite(params.y_grid)),
)

# The Aiyagari decomposition splits wealth dynamics into a
# deterministic-update stage (WealthChange) and a consumption-savings
# argmax (ConsumptionSavings). Closures follow `f(cell, args...; env)`;
# WealthChange passes env as a Ref through its broadcast, so the
# closure unwraps via `env[]`.

wealth_post(cell; env) = (1 + env[].r) * cell.wealth + env[].w * cell.income

income_shock = MarkovAlong(layout; axis = :income, transition = params.P_y)

receipt = WealthChange(layout;
    wealth_post  = wealth_post,
    wealth_axis  = :wealth,
    closure_deps = (:r, :w))

savings = ConsumptionSavings(layout;
    β               = params.β,
    utility         = (cell, c; env) -> (c^(1 - params.σ)) / (1 - params.σ),
    wealth_axis     = :wealth,
    monotone_search = :divide_conquer)   # MPS ≥ 0 from concave u + linear budget

hh = lift_moments(income_shock ∘ₛ receipt ∘ₛ savings;
    K_supplied = at_end(integrand = :wealth, reduce = sum))
```

### 5.2 Outer loop — plain Julia

```julia
function aiyagari_prices(K, p = params)
    r = p.α * (K / p.L)^(p.α - 1) - p.δ
    w = (1 - p.α) * (K / p.L)^p.α
    return (; r, w)
end

# Inner V fixed point: repeat the full Bellman backward until V settles.
function _vfi!(hh, env, V0, kernels, scratches;
               tol = 1e-7, maxiter = 1500)
    V = copy(V0)
    for _ in 1:maxiter
        Vn = backward!(hh, V, env, kernels, scratches)
        d  = maximum(abs, Vn .- V); V .= Vn
        d < tol && return V
    end
    return V
end

# Inner Λ fixed point: forward! does not take env — env was consumed
# by backward! in producing the kernel.
function _lambda!(hh, Λ0, kernels, scratches; tol = 1e-6, maxiter = 20_000)
    Λ = copy(Λ0)
    for _ in 1:maxiter
        Λn = forward!(hh, Λ, kernels, scratches)
        d  = maximum(abs, Λn .- Λ); Λ .= Λn
        d < tol && return Λ
    end
    return Λ
end

# Tatonnement on K.
function aiyagari_steady_state(p = params; K_init = 5.0,
                               update_speed = 0.05, rtol = 2e-2)
    kernels, scratches = allocate(hh)
    dims = layout_size(layout)
    V    = zeros(Float64, dims...)
    Λ    = fill(1.0 / prod(dims), dims...)

    K = K_init
    K_err = Inf
    while abs(K_err) > rtol
        (; r, w) = aiyagari_prices(K, p)
        env = (; K, r, w)
        V   = _vfi!(hh, env, V, kernels, scratches)
        Λ   = _lambda!(hh, Λ, kernels, scratches)
        K_supplied = compute_moments(hh, env).K_supplied
        K_err = (K_supplied - K) / K
        K += update_speed * K_err * K
    end
    (; r, w) = aiyagari_prices(K, p)
    return (; K, r, w, V, Λ)
end
```

What the household-stage code captures: named axes; off-the-shelf
stages configured with model economics (closures + `closure_deps`);
composition with `∘ₛ`; one moment attached via `lift_moments`. What
the consumer writes per-model: the tatonnement / Newton / Picard
outer loop, the production prices, the residual structure. The
library draws a clean line between the two — see §7.

### 5.3 Variants

```julia
# σ-sweep at fixed equilibrium calibration: rebuild the chain with
# the new σ; run the same outer loop. (Param-based env-keyed sweeps
# are also available; see §2.3.)

# Life-cycle: build N_age-replicated chain. Cross-cohort threading
# (bequest, birth, mortality) goes in the consumer's outer-loop code;
# the library does not handle this for you.
hh_age = replicate_age(hh, 60)

# GPU (lift_gpu is documented per stage type but currently raises;
# add a CUDA dep and materialize the methods first).
hh_gpu = lift_gpu(hh)        # not yet runnable

# SSJ at steady state: Steps 2-4 of the fake-news algorithm are in
# HouseholdStages via expectation_vectors / build_F / J_from_F.
# Step 1 (per-period direct effects) is model-specific and lives in
# the consumer's code. See §8.
```

---

## 6. Per-stage lifts and moment attachment

Operational sketches. Theorems (composition preservation, chain-
rule-as-functoriality) live in the companion paper.

### 6.1 Moment attachment (`lift_moments`)

`lift_moments(chain; spec_name = at_end(...))` wraps a chain in a
`MomentedChain` carrier so named aggregates can be read off after a
forward pass via `compute_moments(chain, env)`. The supported spec
is `at_end(integrand, reduce, closure_deps = ())`, end-of-chain only.
The integrand may be a `Symbol` (cell-field shortcut, e.g.
`:wealth`) or a closure `(cell, args...; env) -> value`. The
emitted-moment NamedTuple type is computed at construction from the
moment specs.

Moment attachment is *not* a per-stage lift in the §2.8 sense — it
doesn't produce another stage. The `MomentedChain` is a thin wrapper
around the chain, with backward / forward delegating to the inner
chain and an extra `compute_moments` method that reads the
integrand against the terminal `Λ`.

### 6.2 `lift_gpu`

See §2.7. Rebuilds each stage with `cu(field)` on array-typed static
fields; `allocate` then produces `CuArray` workspace. Methods with
platform-specific kernels dispatch on the buffer's concrete array
type. Per-stage methods currently raise; flesh out when GPU
hardware is on the development machine.

### 6.3 `replicate_age`

`replicate_age(chain, N_age) = product(chain, chain, ..., chain;
axis = :age)`. The household stage operates on a fused `:age`-
indexed tensor; **cross-cohort threading (bequest, birth, mortality)
is the consumer's responsibility**. The library provides the
structural ingredient (the age-indexed chain); the consumer writes
the residual logic.

For age-varying chain content (working vs. retired with different
state spaces), use `product` directly with non-identical components.
Heterogeneous-shape product is currently restricted to uniform
components.

### 6.4 `lift_jacobian`

Two modes:

- **Forward-mode.** Re-allocate workspace as `ForwardDiff.Dual`-
  typed arrays; re-run the existing `backward!` / `forward!`
  methods. Cheap when the number of input directions (e.g., prices
  being perturbed) is small relative to the state space.
- **Reverse-mode.** Per-stage `backward_adjoint!` /
  `forward_adjoint!` methods written alongside the primal. Cheap
  when the number of output directions (e.g., a scalar loss) is
  small relative to the parameter count. **All primitive stages
  have adjoint methods including the choice stages (`Argmax`,
  `LogitChoice`, `Migration`, `ConsumptionSavings`)** — they
  exploit the envelope theorem to reuse the primal-evaluated K
  (stored as the policy / probability tensor in the kernel field)
  as a frozen linear operator for the VJP. Subgradient at boundary
  cells; interior cells are smooth.

Forward-mode handles SSJ Step 1, within-period Newton, and
Euler-equation errors. Both modes can coexist on the same chain.

The chain rule across stages is the **composition preservation** of
`lift_jacobian` — proved in the companion paper. In the library,
this means the chain's own machinery accumulates per-stage Jacobians
without needing a separate "chain-level derivative routine."

---

## 7. Sequence-space utilities

`src/sequence_space.jl` provides three functions that are downstream
consequences of the household-layer abstraction:

```julia
expectation_vectors(chain, integrand, T, kernels, scratches)
build_F(curlyY, curlyD, curlyE)
J_from_F(F)
```

(`expectation_vectors` does not take env: env was consumed by the
prior `backward!` call that populated the `kernels`. The K used for
Kᵀ iteration is the kernel materialized at that backward-pass
evaluation point.)

These implement Steps 2, 3, and 4 of the SSJ fake-news algorithm
(Auclert-Bardóczy-Rognlie-Straub 2021). Step 2 is "iterate the
chain's `forward_adjoint!` to propagate an integrand by `Kᵀ`."
Step 4 is a 7-line anti-diagonal cumulation. Step 1 (per-period
direct effects of a shock) is model-specific and is the consumer's
code.

That the fake-news algorithm reduces to "iterate `forward_adjoint!`
+ cumulate F" is one of the structural payoffs of the K-operator
framing. See the companion paper for the theorem.

---

## 8. Open questions

1. **CPU/GPU kernel split.** KernelAbstractions for elementwise /
   broadcast-shaped kernels (Markov apply, wealth remap)? Hand-rolled
   CUDA for the irregular ones (distribution pushforward,
   monotone-policy argmax)? Or KA everywhere? `lift_gpu` currently
   raises pending this work.

2. **Heterogeneous-shape product.** Currently supports only uniform
   components. Heterogeneous shapes (e.g., working vs. retired with
   different state spaces, age-varying chain content) raise during
   `product(...)` construction. Implementation deferred until a
   consumer needs it.

3. **`Param{T}` swept mode usage.** The library supports
   `Param(:symbol)` for env-keyed parameter sweeps, but no shipping
   consumer uses it yet. The mode is in the library at low cost; if
   no consumer materializes, prune.

4. **Shared outer-loop helpers (revisit).** Patterns repeated across
   consuming models (`_vfi!` / `_lambda!`, tatonnement on `K`,
   damped Picard on vector unknowns) may eventually warrant a shared
   helper module. Current discipline (§7) keeps outer loops
   per-model until a stable pattern emerges across several.
