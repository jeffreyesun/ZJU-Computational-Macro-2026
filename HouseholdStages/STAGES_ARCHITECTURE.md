# Stage architecture — design notes

A design-level companion to the package. For the protocol mechanics
(how to *write* a stage) read `HOWTO_STAGE.md`; this document covers
the *why* — the categorical content, the V/Λ duality, the K-operator
framing, the trichotomy that supports transition paths and eltype
switching.

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

- **`<X>StageSpec`** — pure stage-specific configuration: axis names,
  user closures, parameters. **Layout-free and eltype-free.**
  Immutable by convention. The Spec carries everything the K-operator
  needs besides the runtime `(V_out, env)` and the runtime layout.
- **`StageBuffer{Kernel, Scratch, V, Λ, LIn, LOut}`** — per-call
  state. A single generic struct (in `src/stages/abstract.jl`)
  parameterised on the stage's `Kernel` and `Scratch` types and on the
  concrete V/Λ array types. Carries the materialised K-operator data
  (`kernel`), per-stage scratch (`scratch`), the V/Λ output arrays
  (`V_start`, `Λ_end`), the layouts the buffer was allocated against
  (`input_layout`, `output_layout`), and a `CacheState`. Fresh on
  every `allocate(spec, layout)` call; buffers are not shared across
  uses of the same Spec.
- **`<X>Stage`** — the bundled wrapper `(; spec, buffer)` users
  construct and pass around. Emitted by `@definestage <X>Stage
  <X>StageSpec [kernel=K] [scratch=S]`; only `<X>Stage` is exported,
  Spec is internal.

Under the 2026-05-25 protocol refactor there is no per-stage `Buffer`
struct — every stage shares the generic `StageBuffer`. The Kernel and
Scratch types are the stage's degrees of freedom; everything else
about the buffer is universal.

Most users never see the Spec layer. They construct a
`MarkovStage(layout; axis = :y, transition = P)`, compose with `∘`,
and call `backward!` / `forward!` / `solve_steady_state_given_env!` on
the bundled object. The split exists to make three things tractable:

- **Transition paths.** `solve_transition_given_env_path!` allocates
  `T` per-period chains sharing one Spec, so per-period buffers stay
  separated and the kernel materialised at period `t`'s backward is
  the kernel consumed at period `t`'s forward. The L05 "stale-kernel
  in transition" footgun is impossible by construction.
- **Eltype switching for AD.** `with_eltype(stage, T)` rebundles the
  same Spec against the same layout at the new eltype, allocating a
  fresh Buffer at `T` (e.g., `ForwardDiff.Dual{...}`). Since the Spec
  is layout/eltype-free, this is a single generic operation — no
  per-stage `with_eltype` method needed (see §12).
- **Future GPU/CPU dispatch.** Buffers parametrise on their concrete
  V/Λ array types. A `MarkovStage` whose buffer holds `Matrix` and
  one whose buffer holds `CuArray` are different concrete
  instantiations; algorithmic methods can dispatch on the buffer's
  concrete array type. (See `lift_gpu` in Status.)

Method signatures are **buffer-first at the primary**, with
bundled-Stage delegates:

```julia
backward!(buffer, spec::MarkovStageSpec, V_end, env)  =  ...          # primary
backward!(stage::AbstractStage, V_end, env)           =               # delegate
    backward!(stage.buffer, stage.spec, V_end, env)
```

The buffer-first convention puts dispatchable state (the array types,
the layout types, the kernel struct) at the head of the signature
where multiple dispatch can see it cleanly. The bundled delegate is a
one-liner.

This convention is universal for the per-stage layer — `backward!`,
`forward!`, `backward_adjoint!`, `forward_adjoint!` — and for the
spec-keyed primaries of the outer-loop helpers
(`solve_steady_state_given_env!`,
`solve_transition_given_env_path!`, `compute_direct_jacobian!`),
which live in `outer_loop_internal.jl`. The `outer_loop.jl` file
contains the Stage-keyed public delegates. The split is where the
public Stage-keyed form absorbs nontrivial bookkeeping (warm-starting
from buffer state, copying results back, computing moments); the
per-stage delegates are otherwise one-liners.

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

Layout-changing stages have `S_in ≠ S_out` and the framework tracks
both layouts on the buffer (`input_layout`, `output_layout`). A Spec
declares its output shape by overriding the `output_layout(spec,
layout)` trait (default: identity). `ForgetfulSumStage` overrides it
to `drop_axis(layout, spec.forget_axis)`; backward broadcasts V along
the dropped axis, forward sums Λ along it, and duality holds because
broadcast-along-axis and sum-along-axis are adjoint in the natural
pairing. `input_layout(spec, layout)` exists symmetrically but no
current stage overrides it.

## 4. The stage interface

Every stage class implements (at most):

```julia
allocate_kernel(spec, T, layout)  -> Kernel  # default: nothing
allocate_scratch(spec, T, layout) -> Scratch # default: nothing
backward!(buffer, spec, V_end, env)          -> V_start  # primary
forward!(buffer, spec, Λ_start)              -> Λ_end    # primary
output_layout(spec, layout)                  -> StateLayout  # default: layout
default_eltype(spec)                         -> Type    # default: Float64
static_env_deps(::Type{spec_type})           -> NamedTuple  # default: ()
```

The framework provides `allocate(spec, layout, T)` (builds a fresh
`StageBuffer` by calling `allocate_kernel` / `allocate_scratch`,
sizing `V_start`/`Λ_end` from `input_layout`/`output_layout`), the
bundled wrapper struct + constructors + `bundle` method via
`@definestage`, and the bundled-Stage delegates:

```julia
backward!(stage, V_end, env)                  # 3-arg
forward!(stage, Λ_start)                      # 2-arg, "trusted"
forward!(stage, Λ_start, V_end, env; kwargs)  # 4-arg, cache-checking
```

`backward!` populates `buffer.kernel` (the K-operator's runtime data),
writes into `buffer.V_start`, stamps `buffer.cache` via
`_seat_cache!(buffer, V_end, env)`, and returns `V_start`. `forward!`
reads the kernel, writes `buffer.Λ_end`, returns it. The user reads
layout off the buffer: `(; input_layout, output_layout) = buffer`.

The `_seat_cache!` call is **explicit at the end of every concrete
`backward!`**. Folding it into the bundled-stage delegate was
considered and rejected: the chain backward seats the chain-level
cache after all component backwards have run, and each component
seats its own cache from inside its own backward — both invariants
need to hold simultaneously, and making `_seat_cache!` implicit
would break the symmetric per-component story.

**`forward!` does not take `env`.** Once backward has materialised K
into the kernel, forward needs nothing else. Skipping the env argument
on forward is what makes the cache-checking 4-arg `forward!` possible
(§5) and what eliminates a class of "I changed env between the V and
Λ sweeps and forgot to re-seat the kernel" bugs.

`allocate(spec, layout, T)` produces a fresh Buffer. Pre-allocated
`V_start` and `Λ_end` arrays can be passed as library-internal kwargs
— `ProductStage` uses this to stitch component buffers as views into
a fused tensor. Stages that own all their state (the common case)
never see those kwargs.

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
Base.:∘(a::AbstractStageSpec, b::AbstractStageSpec) = ChainStageSpec(_compose_spec_tuples(a, b))
Base.:∘(a::AbstractStage,     b::AbstractStage)     = ChainStage((a, b))
```

`a ∘ b` is **time-ordered**: `a` runs first. This is the opposite of
Julia's `Base.:∘` on functions, where `f ∘ g` means `x -> f(g(x))`.
The stage convention follows the math (stages composed in time order);
the docstrings on `∘` flag the opposite-of-Function direction.

Composition is a no-allocation operation at the Spec layer:
`ChainStageSpec((s1, s2, …))` holds a tuple of component Specs and a
mutable `moments :: Dict{Symbol, Any}` slot (the only mutable field
on any Spec). Nested `ChainStageSpec`s are auto-flattened: `(a ∘ b) ∘
c` produces a 3-tuple `(a, b, c)`, not a 2-tuple `((a, b), c)`. The
chain's input/output layouts are determined by walking components
through `_allocate_chain_buffers` at allocate time — each component's
`output_layout` is threaded into the next's input. Moments cannot be
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
Spec type. Heterogeneous-type products raise at construction.

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

## 8. `env`, env-resolvable spec fields, closures

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

**Env-resolvable parameters** — stage hyperparameters that may be
calibrated to a literal value or read from `env` at runtime — are
declared as `Union{T, Symbol}` fields:

```julia
struct LogitChoiceStageSpec{F, BF, T<:Real} <: AbstractStageSpec
    choice_axis :: Symbol               # pure-Symbol label field
    ε           :: Union{T, Symbol}     # env-resolvable parameter
    ...
end

# Inside backward!:
ε = resolve(spec.ε, env)   # spec.ε  if Real;  env[spec.ε]  if Symbol
```

The pattern replaces the pre-2026-05-25 `Param{T}` wrapper, which
was deleted. The discriminator is the **declared field type**:
`Union{T, Symbol}` is env-resolvable; pure `Symbol` is a label and
ignored by `effective_env_slice`. `resolve` is type-stable through
union-splitting on the small union. Switching modes — calibrated to
swept — is a Spec rebuild (`<Spec>(...; ε = :ξ)`), not a mutation, so
the change participates cleanly in dispatch.

## 9. Env slicing — `static_env_deps`, `effective_env_slice`

Each concrete Spec type declares a `static_env_deps(::Type{<:X}) ::
NamedTuple` — `env` fields the **type** itself reads, irrespective of
user closures. The default is `NamedTuple()` and no current shipping
stage overrides it. The hook is in place for stages whose Spec
fields themselves reference env keys via mechanisms other than the
`Union{T, Symbol}` pattern.

`effective_env_slice(spec)` is the union of `static_env_deps` and the
runtime env-resolved field names: any `Union{T, Symbol}` field whose
current value is a `Symbol`. The mechanism (`_env_field_names`,
`_is_env_resolvable` in `src/stages/abstract.jl`) iterates fields,
checks the **declared** type (so pure-`Symbol` label fields like
`choice_axis` are correctly skipped — their type is `Symbol`, not a
`Union`), and reads the current `Symbol` value as the env key.

`chain_env_names(chain)` is the union across all component stages.
`env_schema(spec)` returns a prototype NamedTuple of the required env
keys; `make_env(spec; kwargs...)` validates a user-constructed env
against the schema (errors on missing required keys; permissive
about extras).

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
`stage.spec` together with the input layout from the caller's buffer.

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
library ships three.

### `with_eltype(stage, T)` — eltype rebuild

The workhorse. Under the layout/eltype-free Spec, this is a **single
generic operation** — `bundle(stage.spec, stage.buffer.input_layout,
T)` — defined once in `src/lifts/jacobian.jl`. No per-stage method
needed; the same Spec is rebundled against the same layout at the
new T, and the framework's `allocate(spec, layout, T)` does the rest.

`with_eltype` is functorial in `T` (composition of stages and
`with_eltype` commute) and is the foundation of forward-mode AD.

### `lift_jacobian(stage; mode = :forward, n_dual, tag, primal_eltype)`

Forward mode (default): `dual_eltype = ForwardDiff.Dual{tag,
primal_eltype, n_dual}`; rebuild via `with_eltype(stage,
dual_eltype)`. The user typically wraps the rebuilt chain in
`ForwardDiff.derivative` or `ForwardDiff.jacobian`. Static Spec
fields (transitions, costs, user closures) keep their original
eltype; the cross-eltype matmul `Float64 * Dual → Dual` flows through
LinearAlgebra's generic `mul!` fallback.

Reverse mode (`mode = :reverse`): returns the stage unchanged; the
reverse-mode surface is exposed via per-stage adjoint methods. Each
stage in the library has a `forward_adjoint!(spec, dΛ_end, buffer)`
method (and `backward_adjoint!`); choice-stage adjoints (Argmax,
LogitChoice, Migration, ConsumptionSavings) use the envelope theorem
to reuse the K materialised at the primal eval point as a frozen
linear operator for the VJP, with subgradients at tie boundaries.
The pattern of manual per-stage adjoints is flagged as architectural
debt to be redesigned separately (AD-based, structural K-transpose
lift, or other); the current methods are mechanical.

The chain's adjoint passes implement the chain rule: `ChainStageSpec`'s
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

## 14. Adding a new stage

The mechanics — what to write, the `@definestage` line, common
patterns like next-state caching and the shared softmax helper, and a
checklist of footguns — live in **`HOWTO_STAGE.md`**. The protocol
is small: a Spec, optionally a Kernel and Scratch, `allocate_kernel`
/ `allocate_scratch`, `backward!`, `forward!`, and one
`@definestage` line.

The single most useful correctness test for a new stage is the
**duality identity** `⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩`. If
it holds, the stage's `forward!` and `backward!` are truly adjoint,
which catches most off-by-one indexing errors, share-vs-sum-along-axis
mismatches, and policy-vs-action-index confusions in one number.

Canonical worked examples, in ascending complexity:
`identity_stage.jl` (21 LOC), `utility.jl` (28 LOC), `markov_along.jl`
(85 LOC), `logit_choice.jl` (85 LOC), `argmax.jl` (92 LOC). Read
alongside `HOWTO_STAGE.md`.

## 15. Status

- **`lift_gpu` is not yet implemented.** Entry points
  (`lift_gpu(::AbstractStageSpec)` and `lift_gpu(::AbstractStage)`)
  raise. The path forward: add CUDA as an optional / extension dep,
  define per-stage Spec methods that rebuild with `cu(field)` on
  array-typed static fields, let algorithm-divergent methods
  dispatch on the buffer's concrete array type. The Spec/Buffer
  trichotomy is already structured to support this — the generic
  `StageBuffer` parametrises on its V/Λ array types, so a chain
  whose buffer holds `CuArray`-typed slots is a different concrete
  instantiation of the same struct definition.
- **`WealthChangeStage.backward_adjoint!` is stubbed.** Only
  `forward_adjoint!` is implemented (which is what
  `expectation_vectors` needs). The backward adjoint would mirror
  the share-based gather as a scatter — add when reverse-mode
  gradients through V on this stage are needed.
- **Per-stage adjoints are flagged for redesign.** The manual
  `forward_adjoint!` / `backward_adjoint!` methods are mechanical
  ports of the primal operators and represent architectural debt.
  Candidates for the redesign include AD-driven adjoints, a
  structural K-transpose lift, or generated adjoints from the
  K-operator declaration. Do not extend the manual pattern to new
  stages without surfacing the design question first.
- **`static_env_deps` has no current overriders.** Every concrete
  Spec returns `NamedTuple()`. With closure env reads not
  introspected, the hook is scaffolding for stages whose Spec
  fields themselves reference env keys via mechanisms other than
  the `Union{T, Symbol}` env-resolvable pattern. Kept as a
  documented extension point.
- **`ProductStage` v1 requires uniform components** (same concrete
  Spec type). Heterogeneous shapes (working vs. retired with
  different state spaces, age-varying chain content) raise at
  construction. Implementation deferred until a consumer needs it.
- **`compute_direct_jacobian!` is diagonal-only.** Period-0 direct
  effect on `J[t, t]`, off-diagonals zero. The real fake-news
  Jacobian goes through `expectation_vectors + build_F + J_from_F`
  (§13). The function name was chosen to advertise the v1 scope.
