# Stage architecture — design notes

## 1. Overview — what a stage is

A **stage** is a configuration struct that, given a continuation value
function `V_out` over an *output* state space `S_out` and an
environment `env`, produces a linear operator `K : M(S_in) → M(S_out)`
on measures, a flow payoff `r ∈ V(S_in)`, and the corresponding
backward / forward sweep:

```
backward:  V_in   =  r + Kᵀ V_out
forward:   Λ_out  =  K Λ_in
```

The operator `K` is the **kernel**. Different stage classes
materialise `K` differently — sometimes K is the configuration itself
(MarkovStage: K is just the transition matrix), sometimes K is a small
per-cell datum populated at backward time (ArgmaxStage: K is an
integer policy; LogitChoiceStage: K is a per-cell action probability
tensor; WealthChangeStage: K is the materialised wealth-post array
plus an implicit linear-interpolation rule). The kernel slot stores
whatever small thing parameterises K; the chain doesn't materialise K
as a dense matrix.

Stages compose under `∘` (sequential composition in time) and product
under `×` (parallel composition across components). `ChainStage` and
`ProductStage` are themselves stages — closure under both operators
means any compound built from primitives is itself a stage and
inherits the same interface.

## 2. The Spec / Buffer / Stage trichotomy

Each concrete stage class — `MarkovStage`, `ArgmaxStage`,
`WealthChangeStage`, … — factors into three layers:

- **`<X>StageSpec`** — pure configuration: layout, transition matrix,
  closures, `Param`-typed hyperparameters, `element_type`. Immutable
  by convention. The Spec carries everything the K-operator needs
  besides the runtime `(V_out, env)`.
- **`<X>StageBuffer`** — per-call state: `kernel`, `scratch`,
  `V_start`, `Λ_end`, `cache::CacheState`. Fresh on every
  `allocate(spec)` call; buffers are not shared across uses of the
  same Spec. A transition path's per-period chains share one Spec but
  each carry their own Buffer.
- **`<X>Stage`** — the bundled wrapper `(; spec, buffer)` users
  construct and pass around. Only `<X>Stage` is exported; Spec and
  Buffer are internal.

Most users never see the Spec/Buffer layer. They construct a
`MarkovStage(layout; axis = :y, transition = P)`, compose with `∘`,
and call `backward!` / `forward!` / `solve_steady_state_given_env!` on
the bundled object. The split exists to make three things tractable:

- **Transition paths.** `solve_transition_given_env_path!` allocates
  `T` per-period chains sharing one Spec, so per-period buffers stay
  separated and the kernel materialised at period `t`'s backward is
  the kernel consumed at period `t`'s forward. The L05 "stale-kernel
  in transition" footgun is impossible by construction.
- **Eltype switching for AD.** `with_eltype(stage, T)` rebuilds the
  Spec under a new element type (e.g., `ForwardDiff.Dual{...}`) and
  bundles a fresh Buffer at that eltype. The user's closures are
  shared across eltypes; static array fields can be promoted with
  conversion; the Spec stays small.
- **Future GPU/CPU dispatch.** Spec and Buffer carry concrete array
  types as parametric type variables. A `MarkovStage{Matrix, ...}`
  and a `MarkovStage{CuArray, ...}` are different concrete
  instantiations; algorithmic methods can dispatch on the buffer's
  concrete array type. (See `lift_gpu` in Status.)

Method signatures are **Spec/Buffer-keyed at the primary**, with
bundled-Stage one-line delegates:

```julia
backward!(spec::MarkovStageSpec, V_end, env, buffer)  =  ...           # primary
backward!(stage::AbstractStage,  V_end, env)          =                # delegate
    backward!(stage.spec, V_end, env, stage.buffer)
```

This convention is universal: `backward!`, `forward!`,
`backward_adjoint!`, `forward_adjoint!`, `allocate`, `with_eltype`,
`solve_steady_state_given_env!`, `solve_transition_given_env_path!`,
`compute_direct_jacobian!` — each has a Spec/Buffer-keyed primary in
the stage-implementation files (or `outer_loop_internal.jl` for the
solver helpers) and a Stage-keyed delegate in the public surface
(`outer_loop.jl`). Julia's multiple dispatch routes the calls.

The `outer_loop.jl` / `outer_loop_internal.jl` split is the only place
the public Stage-keyed form absorbs nontrivial bookkeeping
(warm-starting from buffer state, copying results back, computing
moments). The delegate is otherwise a one-liner.

## 3. The K-operator and V/Λ duality

State spaces are products of named axes (§7). Over a state space `S`
live two function spaces:

- `V(S)` — real-valued functions (value functions).
- `M(S)` — signed measures (distributions).

Pair them by `⟨V, Λ⟩ = ∫_S V dΛ`. Aggregates / moments are linear
functionals `M(S) → ℝ`.

A stage's K-operator is in `Hom(M(S_in), M(S_out))`. Its adjoint `Kᵀ`
sits in `Hom(V(S_out), V(S_in))`. Backward applies `Kᵀ` to the value
function; forward applies `K` to the distribution. Adjointness gives
the duality identity

```
⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩
```

where `r` is the stage's flow payoff (often zero — only `UtilityStage`
and `ConsumptionSavingsStage` contribute interesting flow payoffs in
the current catalog; for everything else the identity simplifies to
the unweighted form). Every stage that defines a K-operator inherits
this identity for free. The test suite exercises it as the per-stage
correctness check — duality holds iff backward and forward apply
truly adjoint operators.

Layout-changing stages (`ForgetfulSumStage` drops an axis) have
`S_in ≠ S_out`: backward broadcasts V along the dropped axis, forward
sums Λ along it, and duality holds because broadcast-along-axis and
sum-along-axis are adjoint in the natural pairing.

## 4. The stage interface

Every stage implements four Spec-keyed primaries:

```julia
allocate(spec, T = spec.element_type)         -> Buffer
backward!(spec, V_end, env, buffer)           -> V_start
forward!(spec,  Λ_start, buffer)              -> Λ_end
static_env_deps(::Type{spec_type})            -> NamedTuple  # default: ()
```

Bundled `AbstractStage` delegates:

```julia
backward!(stage, V_end, env)                  # 3-arg
forward!(stage, Λ_start)                      # 2-arg, "trusted"
forward!(stage, Λ_start, V_end, env; kwargs)  # 4-arg, cache-checking
```

`backward!` populates `buffer.kernel` (the K-operator's runtime data),
writes into `buffer.V_start`, stamps `buffer.cache` via
`_seat_cache!`, and returns `V_start`. `forward!` reads the kernel,
writes `buffer.Λ_end`, returns it.

**`forward!` does not take `env`.** Once backward has materialised K
into the kernel, forward needs nothing else. Skipping the env argument
on forward is what makes the cache-checking 4-arg `forward!` possible
(§5) and what eliminates a class of "I changed env between the V and
Λ sweeps and forgot to re-seat the kernel" bugs.

`allocate(spec, T)` produces a fresh Buffer. Pre-allocated `V_start`
and `Λ_end` arrays can be passed as kwargs — `ProductStage` uses this
to stitch component buffers as views into a fused tensor:

```julia
allocate(spec, T; V_start = nothing, Λ_end = nothing) -> Buffer
```

Stages that own all their state (the common case) ignore the kwargs.

## 5. The kernel cache

Every Buffer carries a `CacheState`:

```julia
mutable struct CacheState
    last_V_hash  :: UInt
    last_env     :: Any        # NamedTuple or nothing
    kernel_valid :: Bool
end
```

`backward!` stamps the cache (`_seat_cache!(buffer, V_end, env)`)
after populating the kernel. The cache fingerprint is `(hash(V_end),
env)` plus a validity flag.

The bundled `AbstractStage` exposes a cache-checking `forward!`:

```julia
forward!(stage, Λ_start, V_end, env;
         reseat_if_stale = false, check = true)
```

This compares `(hash(V_end), env)` against the buffer's cache. On a
match it runs the trusted 2-arg forward; on a mismatch it either
errors with a useful message (default) or re-runs `backward!` first
(`reseat_if_stale = true`). The `check = false` opt-out is for
power users who know the kernel is fresh and want to skip the hash.

The chain-level cache lives on `chain.buffer.cache` (seated by the
generated chain backward at the chain level) **and** each component's
buffer cache is seated independently inside the per-component
`backward!`. Both invariants hold simultaneously: a user can
cache-check at the chain level or at any per-stage level. Cache
invalidation propagates: `invalidate!(chain.buffer)` iterates over
the component buffers.

`solve_transition_given_env_path!` doesn't need cache checking — it
allocates a fresh per-period Buffer for each `t`, so the kernel
materialised at period `t`'s backward is by construction the kernel
read at period `t`'s forward. The cache machinery is for users who
hand-roll transition or perturbation drivers and would otherwise have
to manage the invariant themselves.

## 6. Composition under `∘`, product under `×`

```julia
Base.:∘(a::AbstractStageSpec, b::AbstractStageSpec) = ChainStageSpec((a, b))
Base.:∘(a::AbstractStage,     b::AbstractStage)     = bundle(a.spec ∘ b.spec)
```

`a ∘ b` is **time-ordered**: `a` runs first. This is the opposite of
Julia's `Base.:∘` on functions, where `f ∘ g` means `x -> f(g(x))`.
The stage convention follows the math (stages composed in time order);
the docstrings on `∘` flag the opposite-of-Function direction.

Composition is a no-allocation operation at the Spec layer:
`ChainStageSpec((s1, s2, …))` holds a tuple of component Specs, a
mutable `moments :: Dict{Symbol, Any}` slot (the only mutable field
on any Spec), and an `out_layout` derived from the last component.
Nested `ChainStageSpec`s are auto-flattened: `(a ∘ b) ∘ c` produces a
3-tuple `(a, b, c)`, not a 2-tuple `((a, b), c)`. Moments cannot be
attached before composition: `∘` refuses to compose a chain that
already carries moments (call `define_moment!` last).

`ChainStage`'s `backward!` and `forward!` iterate over the component
tuple via `@generated`. The naive runtime loop `for i in n:-1:1` over
a heterogeneous tuple is type-unstable (each slot has a different
concrete Spec type); the `@generated` unroll recovers full type
stability and brings chain backward+forward to within ~1.1×–1.2× of
hand-coded reference kernels at example sizes.

`s1 × s2` builds a `ProductStageSpec` along a new axis (default
`:group`). The product's K is the block-diagonal direct sum of the
component K's; backward and forward operate per-component on slices
of a fused tensor. The component buffers' `V_start` / `Λ_end` are
**views** into the fused tensors, so per-component sweeps write
directly into the right slice with no copies. Asymmetric-layout
components (where the component's output layout differs from its
input layout — e.g., a `ForgetfulSumStage` inside a product) keep
the V-side view but fall back to a fresh `Λ_end` allocation; the
`_accepts_view_Λ` predicate handles the dispatch.

`replicate_age(stage, N; axis = :age)` is sugar for
`product(stage, stage, …, stage; axis)`. The library does not handle
cross-cohort threading (bequest, birth, mortality) — that's the
caller's responsibility, structurally identical to time-fixed-point
threading.

v1 of `ProductStageSpec` requires uniform components: same concrete
Spec type, same input layout (via cheap structural equality on axis
names and sizes). Heterogeneous-shape products raise at construction.

## 7. Layouts, axes, cell iteration

```julia
layout = StateLayout(
    StateAxis(:wealth,   continuous_grid(0, 100; length = 400, spacing = :log)),
    StateAxis(:income,   discrete_finite([0.6, 1.0, 1.4])),
    StateAxis(:location, categorical([:home, :abroad])),
)
```

Two axis kinds:

- `continuous_grid(grid)` or `continuous_grid(lo, hi; length, spacing = :linear | :log)` —
  a numeric grid for interpolation.
- `discrete_finite(levels)` — a finite vector of levels. Leaf type
  is any bits-typed value (`Float64`, `Int`, `Symbol`).
  `categorical(syms)` is sugar for `discrete_finite([:s1, :s2, …])`.

`StateAxis(name, kind)` is the canonical form; `StateAxis(name, vec)`
is a shortcut that wraps a raw `AbstractVector` in `discrete_finite`.

Axis names live in the `StateLayout` type parameter so NamedTuple-keyed
iteration is type-stable. `layout_size(layout)` returns the per-axis
sizes; `axis_position(layout, :name)` returns the integer dimension
number; `axisvalues(axis)` returns the grid or level vector.

`cells(layout)` is the canonical iteration form, yielding `(idx,
cell)` pairs:

```julia
for (idx, cell) in cells(layout)
    # idx  :: NamedTuple of integer axis indices
    # cell :: NamedTuple of axis values
end
```

`cell_array(layout)` returns an N-D `Array{NamedTuple}` of cell values
— useful for broadcasting closures over all cells:

```julia
mask = some_predicate.(cell_array(layout); env)
```

(Closures broadcast with `env` as a kwarg are captured per-broadcast,
not broadcast themselves.)

## 8. `env`, `Param`, closures

`env` is the household's economic environment — a `NamedTuple`
carrying prices, calibration scalars, aggregate-state coordinates,
swept-parameter values. It's the only runtime configuration passed to
`backward!`; the chain shares one `env` across all stages.

User closures (`utility`, `wealth_post`, `flow_payoff`,
`next_state_idx`, moment integrands, borrowing-constraint predicates,
migration amenity shifters) follow one signature convention:

```julia
f(cell, positional_args...; env) -> value
```

`cell` first, any stage-specific positional arguments next (`c` for
consumption, `action` for choice, `destination` for migration
amenity), `env` as a kwarg. The kwarg form makes `env` get captured
as a scalar through broadcasts (no `Ref` wrapping needed) and lets
the closure read its env fields plainly: `env.r`, `env.w`.

Closures are stored on the Spec as a typed field (e.g., `utility ::
F_u` with `F_u` a type parameter). The closure type participates in
the Spec's type signature, so dispatch on different closures
specialises the hot path.

`Param{T}` wraps a stage hyperparameter that can be calibrated or
swept at runtime:

```julia
ε = Param(0.25)                 # calibrated: literal value
ε.val = :ξ_logit                 # mode-flip: now reads env.ξ_logit
```

`resolve(p, env)` is type-stable through union-splitting on the
small `Union{T, Symbol}`. Swept Params register in the Spec's
effective env slice (§9). Mode-flip is a field mutation (Param is
mutable), not a stage rebuild — useful for estimation outer loops and
sensitivity sweeps. (The pattern is supported but not exercised by
any shipping example; see Status.)

## 9. Env slicing — `static_env_deps`, `effective_env_slice`

Each concrete Spec type declares a `static_env_deps(::Type{<:X}) ::
NamedTuple` — `env` fields the **type** itself reads, irrespective of
user closures. The default is `NamedTuple()` and no current shipping
stage overrides it (closure-borne env reads are the dominant pattern
and aren't introspected). The hook is in place for stages whose Spec
fields themselves reference env keys — e.g., a future `RateChange`
stage that statically reads `env.r_path` without a user closure.

`effective_env_slice(spec)` is the union of `static_env_deps` and any
swept `Param` keys. `chain_env_names(chain)` is the union across all
component stages. `env_schema(spec)` returns a prototype NamedTuple of
the required env keys; `make_env(spec; kwargs...)` validates a
user-constructed env against the schema (errors on missing required
keys; permissive about extras).

Closure-borne env reads are **not** introspected. Missing fields
surface at the first `backward!` call as `getproperty` errors. The
deliberate trade-off: closure-AST analysis is brittle, and the runtime
error message is informative enough that this hasn't bitten in
practice.

## 10. Moments

`define_moment!(chain, name, spec)` attaches a named moment to the
chain's Spec (the `moments` dict — the only mutable Spec field).
`define_moments!(chain; kwargs...)` is the batch form. Both are
append-only by default; `overwrite_existing_moment_definitions = true`
opts in to overwriting.

Moments are `MomentSpec` structs constructed via `at_end`:

```julia
at_end(; integrand, reduce, aggregate_over = nothing,
         weights = nothing, per = nothing)
```

`integrand` is either a closure `(cell, args...; env) -> value` or a
`Symbol` cell-field shortcut (`:wealth` is sugar for `(cell; env) ->
cell.wealth`). `reduce` is typically `sum` or `mean`.
`aggregate_over` / `per` (both `Symbol` or `nothing`) select a product
axis to reduce over or split along; `weights` (`Symbol` or `nothing`)
names an env field or per-cell weight (`nothing` means Λ is the
weight).

`compute_moments(chain, Λ, env)` evaluates all attached moments
against `Λ`. The signature is non-mutating and takes `Λ` explicitly
— it doesn't read buffer state. Per-cell integrand evaluation goes
through `cell_array(layout)` so the broadcast captures `env` cleanly.

## 11. User-facing solvers

`outer_loop.jl` provides the Stage-keyed public solvers;
`outer_loop_internal.jl` provides the Spec/Buffer-keyed primitives.
The public Stage-keyed `solve_steady_state_given_env!` absorbs four
pieces of bookkeeping the internal primitive deliberately omits:

1. **Warm-start `V` from buffer state** if non-zero (so successive
   calls at perturbed env reuse the previous solution).
2. **Default `Λ_init`** to the uniform distribution if the user
   doesn't provide one. Λ converges fast from uniform, and the
   buffer's Λ_end slot can carry half-iterated state from a prior
   forward pass.
3. **Write converged V and Λ back into the buffer** for the next
   call's warm start.
4. **Evaluate `compute_moments`** if the chain has any moments
   attached; empty NamedTuple otherwise.

The internal primitive `solve_steady_state_given_env!(spec, env,
buffer; ...)` does the iteration only. Both share the same function
name; Julia's multiple dispatch routes them. Stage-level callers
typically want the public form; lift authors building on the spec
layer use the primitive directly.

`solve_transition_given_env_path!` is structurally different — it
allocates `T` per-period chains internally and runs backward then
forward sweeps across the path. There is no single Buffer for the
caller to thread (per-period buffers are an implementation detail), so
the Stage-keyed form is a thin delegate that just forwards
`stage.spec`.

`compute_direct_jacobian!` is a diagnostic helper, **direct-effect
only**: period-0 finite-difference of moments with respect to named
env fields, written on the diagonal of a `T × T` matrix. Off-diagonal
entries are zero by construction. The function name advertises the
v1 scope — the real fake-news Jacobian goes through
`expectation_vectors + build_F + J_from_F` (§13).

## 12. Lifts

A **lift** takes a stage and produces a new stage of the same kind
(with possibly different eltype, array type, or dimensionality),
preserving the K-operator structure and hence composition. The
library ships four:

### `with_eltype(spec_or_stage, T)` — eltype rebuild

The workhorse. Returns a new Spec with `element_type = T` and `Param`
fields re-typed; the bundled-stage form (`with_eltype(stage, T)`)
returns a fresh `<X>Stage` whose buffer is allocated at the new
eltype. Each concrete Spec implements its own method
(`src/lifts/jacobian.jl`). `ChainStageSpec` and `ProductStageSpec`'s
`with_eltype` delegate to their components and preserve moments.

`with_eltype` is functorial in `T` (composition of stages and
`with_eltype` commute) and is the foundation of forward-mode AD.

### `lift_jacobian(stage; mode = :forward, n_dual, tag, primal_eltype)`

Forward mode (default): `dual_eltype = ForwardDiff.Dual{tag,
primal_eltype, n_dual}`; rebuild via `with_eltype(stage,
dual_eltype)`. The user typically wraps the rebuilt chain in
`ForwardDiff.derivative` or `ForwardDiff.jacobian`. Static fields
(transitions, costs, user closures) keep their original eltype; the
cross-eltype matmul `Float64 * Dual → Dual` flows through
LinearAlgebra's generic `mul!` fallback.

Reverse mode (`mode = :reverse`): returns the stage unchanged; the
reverse-mode surface is exposed via per-stage adjoint methods. Each
stage in the library has a `forward_adjoint!` method; choice-stage
adjoints (Argmax, LogitChoice, Migration, ConsumptionSavings) use the
envelope theorem to reuse the K materialised at the primal eval point
as a frozen linear operator for the VJP, with subgradients at tie
boundaries.

The chain's adjoint passes implement the chain rule: `ChainStage`'s
`backward_adjoint!` iterates over components in forward order;
`forward_adjoint!` iterates in reverse. Both read from `spec.stages[i]`
and `buffer.stages[i]` in lockstep.

### `lift_gpu(stage)` — scaffolded

Not yet implemented. The entry points raise. See Status.

### `replicate_age(stage, N; axis = :age)`

`product(stage, stage, …, stage; axis)`. The product axis defaults to
`:age` here (vs. `:group` for raw `product`). Useful for life-cycle
problems where N identical cohorts coexist; cross-cohort threading is
the caller's responsibility.

## 13. Sequence-space utilities

`src/sequence_space.jl` implements Steps 2–4 of the SSJ fake-news
algorithm. Step 1 (per-period direct effects of a shock) is
model-specific.

```julia
expectation_vectors(chain, integrand, T) -> Vector{Array}
build_F(curlyY, curlyD, curlyE)          -> Matrix
J_from_F(F)                              -> Matrix
```

`expectation_vectors` iterates the chain's `forward_adjoint!` (K-
transpose action on a per-cell integrand) for `t = 0, 1, …, T − 1`.
The chain's kernels must have been seated by a prior `backward!` at
the steady-state env — no env argument here, because K was
materialised at the prior backward's eval point.

That the fake-news algorithm reduces to "iterate `forward_adjoint!`
and cumulate F" is one of the structural payoffs of the K-operator
framing. The companion paper proves the equivalence; the library
realises it as code.

`examples/aiyagari_mit_shock/ssj.jl` runs the full pipeline on the
3-stage Aiyagari chain.

## 14. Conventions for stage authors

If you're adding a new stage `<Y>Stage`, the checklist:

1. **Define three structs.** `<Y>StageSpec <: AbstractStageSpec` with
   the immutable configuration. `<Y>StageBuffer <: AbstractStageBuffer`
   with `kernel`, `scratch`, `V_start`, `Λ_end`, `cache::CacheState`.
   `<Y>Stage <: AbstractStage` with `(; spec, buffer)`.
2. **Spec constructor.** `<Y>StageSpec(layout; kwargs...)` resolves
   axis positions, wraps any `Param`-typed kwargs (`p isa Param ? p :
   Param(p)`), validates shapes. Use `Val`-typed type parameters for
   compile-time options like `extrap` on WealthChange or
   `monotone_search` on ConsumptionSavings.
3. **`<Y>Stage` constructor.** Takes the same kwargs as the Spec plus
   optional `V_start` / `Λ_end` (for view-stitching by ProductStage).
   Builds the Spec, allocates a Buffer, returns the bundled wrapper.
4. **`allocate(spec, T; V_start, Λ_end)`.** Allocate `V_start`,
   `Λ_end`, kernel-shaped state, scratch, fresh `CacheState`. Use
   `_alloc_VΛ(layout, T, V_start, Λ_end)` to handle the
   optional-passthrough pattern.
5. **`backward!(spec, V_end, env, buffer)`.** Populate
   `buffer.kernel` from `(V_end, env)`, write `buffer.V_start`, call
   `_seat_cache!(buffer, V_end, env)`, return `V_start`.
6. **`forward!(spec, Λ_start, buffer)`.** Read `buffer.kernel`,
   write `buffer.Λ_end`, return it. Do not consult `env` — env was
   consumed by backward.
7. **`static_env_deps(::Type{<Y>StageSpec})`** if the stage type
   reads env fields not declared by user closures. Default:
   `NamedTuple()`. Most stages don't need to override.
8. **`with_eltype(spec, T)`** — rebuild the Spec with the new
   `element_type` and re-typed `Param` fields. One-liner that
   reconstructs the Spec via its kwarg constructor.
9. **`bundle(spec) = <Y>Stage(spec)`.** One-liner; mirrors every
   other concrete stage.
10. **Reverse-mode adjoints if applicable.** If the K-operator is
    V/θ-independent (linear K), `forward_adjoint!` and
    `backward_adjoint!` follow from primal apply. For non-linear-K
    stages, use the envelope theorem at the materialised K (see
    `LogitChoiceStage`'s adjoint methods for the template).
11. **Tests.** Per-stage backward/forward correctness; duality
    identity (`⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩`);
    composition associativity with existing stages; ForwardDiff
    rebuild via `lift_jacobian(stage; mode = :forward)`; adjoint
    correctness via finite differences against the primal.

The duality identity (item 11) is the single most useful correctness
test. If it holds, the stage's `forward!` and `backward!` are truly
adjoint, which catches most off-by-one indexing errors, share-vs-
sum-along-axis mismatches, and policy-vs-action-index confusions in
one number.

## 15. Status

- **`lift_gpu` is not yet implemented.** Entry points
  (`lift_gpu(::AbstractStageSpec)` and `lift_gpu(::AbstractStage)`)
  raise. The path forward: add CUDA as an optional / extension dep,
  define per-stage Spec methods that rebuild with `cu(field)` on
  array-typed static fields, let algorithm-divergent methods
  dispatch on the buffer's concrete array type. The Spec/Buffer
  trichotomy is already structured to support this — buffers
  parametrise on their concrete array type, so a
  `MarkovStage{CuArray,...}` is a different concrete instantiation
  of the same struct definition.
- **`WealthChangeStage.backward_adjoint!` is stubbed.** Only
  `forward_adjoint!` is implemented (which is what
  `expectation_vectors` needs). The backward adjoint would mirror
  the share-based gather as a scatter — add when reverse-mode
  gradients through V on this stage are needed.
- **`Param`-keyed swept mode is supported but unexercised by any
  shipping example.** The pattern is `Param(:env_key)` for a field
  that reads its value from `env.env_key` at evaluation time. If no
  consumer materialises in 2–4 weeks, candidate for pruning.
- **`static_env_deps` is a hook with no current overriders.**
  Every concrete Spec returns `NamedTuple()`. With `closure_deps`
  dropped (closures' env reads are not introspected), the hook is
  scaffolding for stages whose Spec fields themselves reference env
  keys. Kept as a documented extension point.
- **`ProductStage` v1 requires uniform components.** Heterogeneous
  shapes (working vs. retired with different state spaces,
  age-varying chain content) raise at construction. Implementation
  deferred until a consumer needs it.
- **`compute_direct_jacobian!` is diagonal-only.** Period-0 direct
  effect on `J[t, t]`, off-diagonals zero. The real fake-news
  Jacobian goes through `expectation_vectors + build_F + J_from_F`
  (§13). The function name was chosen to advertise the v1 scope.
