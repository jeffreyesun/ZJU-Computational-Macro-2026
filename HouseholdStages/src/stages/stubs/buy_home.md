# `BuyHome` stub

> Backend gap. Not implemented in `HouseholdStages/` v1. This note
> captures what the reference implementation does and what the library
> would need to support before this stage can be added as a first-class
> `AbstractStage`.

## What `buy_home` does

From `reference_materials/example_stages/buy_home.jl`:

- **Backward.** Existing homeowners (`h ≥ 2`) keep `V_postbuy`. Renters
  (`h = 1`) take `max V_postbuy` over their available housing-size
  choices: `V_choosebuy[:,:,1,:] = maximum(V_postbuy; dims = H_DIM)`.
- **Forward.** Renters scatter across housing-size choices according to
  a softmax `P_buy = exp(ξ V_postbuy) / Σ_{h'} exp(ξ V_postbuy)`;
  existing homeowners pass through unchanged.

The backward currently takes the `max` (a hard argmax over h-choices
available to renters). The forward uses a logit. The two should agree
under `taste_shocks = true` once the backward is updated to be the
log-sum-exp (the codebase has a `#TODO Compute P_buy in backward pass`
noting this).

## Why `LogitChoiceStage` doesn't fit as-is

The library's `LogitChoiceStage` stage is a *uniform* logit over the full
choice axis at every cell. `buy_home` is a **gated** logit: only the
`h = 1` slice of cells faces the choice; cells with `h ≥ 2` pass
through unchanged.

You *can* encode the gate inside the `flow_payoff` closure (`-Inf` for
unavailable actions, return `cell.h` as the only available action for
homeowners), but:

1. The forward pass still iterates *every* cell rather than just the
   gated subset. Wasted work.
2. The semantics of "unavailable action" inside `LogitChoiceStage` is
   ad-hoc (currently: `-Inf` flow payoffs are skipped in the
   log-sum-exp, see `src/stages/logit_choice.jl:90-114`). It works,
   but the *gated logit* pattern is structurally cleaner than the
   `-Inf` workaround.

## What the library would need

Two reasonable approaches:

### Option A — `GatedLogitChoice`

A new stage that takes a per-cell "available" predicate `available(cell;
env) -> Bool` *and* the usual `LogitChoiceStage` parameters. Cells where
`available == false` are pass-through identity; cells where `available
== true` run the standard logit. The kernel stores a gate mask + the
choice-probability tensor restricted to gated cells.

API sketch:

```julia
BuyHome = GatedLogitChoice(layout;
    choice_axis     = :h,
    flow_payoff     = (cell, h_choice; env) -> 0.0,  # no explicit u(buy)
    next_state_idx  = (cell, h_choice) -> h_choice,
    available       = (cell; env) -> cell.h == h_grid[1],  # renters only
    ε               = Param(1.0 / params.ξ),
    closure_deps    = (...),
)
```

Composition with the existing `LogitChoiceStage` is structurally the same;
the forward kernel needs to know which cells to scatter from.

### Option B — `LogitChoiceStage` with first-class action-availability

Add an `available::PayoffFn` field to `LogitChoiceStage` itself (default:
"all actions available"). The backward log-sum-exp consults
`available(cell, action; env)` and skips unavailable actions; the
forward respects the same mask. The "gated" case `buy_home` then drops
out by setting `available = (cell, action; env) -> (cell.h == 1)`
(every action is "unavailable" for homeowners, who effectively pass
through).

The wrinkle: when *no* action is available at a cell, the current
implementation `error`s. We'd need a fallback (identity?) for the
gated case.

## Recommendation when this is built

Lean toward Option A. The gated-logit pattern recurs in HA macro (firm
entry/exit, durable-good purchase, retirement timing), and a dedicated
stage type reads more clearly in chains than a `LogitChoiceStage` with a
non-trivial `available` predicate that the reader has to peer into.
Option A also keeps `LogitChoiceStage` lean for the cases where every
action is genuinely available.

## Tests it would need

- Backward log-sum-exp matches a hand-built `LogitChoiceStage` on the gated
  subset; identity elsewhere.
- Forward scatter matches the same logit on the gated subset; identity
  elsewhere.
- Duality identity on the gated subset.
- Composition with `IdentityStage` on either side preserves behaviour.
- AD: forward-mode (rebuild with `Dual` eltype) gives the right
  Jacobian.
