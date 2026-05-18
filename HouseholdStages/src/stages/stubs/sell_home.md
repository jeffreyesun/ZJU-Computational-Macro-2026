# `SellHome` stub

> Backend gap. Not implemented in `HouseholdStages/` v1. This note
> captures what the reference implementation does and what the library
> would need before this stage can be added.

## What `sell_home` does

From `reference_materials/example_stages/sell_home.jl`:

- **Backward.** For homeowners (`h ≥ 2`), compute
  `V_sell = apply_wealth_change_V!(V_sell_postsell, wealth_postsell_k_presell)`
  where `wealth_postsell = wealth_presell - ϕ·q·h` (realtor fee `ϕ`).
  Then `V_choosesell = max(V_nosell, V_sell)`.
- **Forward.** Logit choice between selling and not, with `P_sell =
  exp(ξ V_sell) / (exp(ξ V_sell) + exp(ξ V_nosell))` per cell. Sellers
  move to the renter slice `h = 1` after applying the wealth change.

This stage is fundamentally **three operations fused**:

1. A wealth change (cost of selling: subtract `ϕ·q·h`).
2. A logit choice over `{sell, keep}` per homeowner cell.
3. An axis-collapse on `:h` — sellers all land at `h = 1`.

## What's missing in the library

Each of the three operations has a primitive in the library:

- (1) Wealth change → `WealthChange` (closure-based).
- (2) Logit choice over `{sell, keep}` → `LogitChoice` (categorical
  choice axis).
- (3) Axis collapse on `:h` → not directly; `ForgetfulSum` drops an
  axis entirely, but here we want "homeowners map to `h = 1`," which
  is a *partial* collapse depending on the choice outcome.

A naïve split `WealthChange ∘ₛ LogitChoice(:keep_or_sell) ∘ₛ <collapse>`
loses the fusion: the wealth change applies *only to sellers* (not to
non-sellers), so it has to be conditional on the choice. The library
doesn't currently have a way to make stage (1) conditional on the
forward-time outcome of stage (2).

Note that the reference implementation itself has this incomplete:
the file carries `#TODO Apply wealth change to sellers` — the wealth-
change-for-sellers step isn't actually applied in the codebase's
forward pass. So this stage isn't even feature-complete in the
reference; it's an open piece of the housing model.

## What the library would need

The natural primitive is a **choice-with-state-transition** stage:

```
ChoiceWithTransition(layout;
    choice_axis,
    flow_payoff(cell, action; env),
    state_transition(cell, action; env) -> NamedTuple of new field values,
    ε,
    closure_deps,
)
```

where each action carries *its own* state-transition rule. For
`sell_home`:

- `action = :keep`: `state_transition(cell; env) = (;)` (no change).
- `action = :sell`: `state_transition(cell; env) = (; h = 1, wealth = cell.wealth - env.ϕ * env.q * cell.h)`.

The backward pass evaluates `r(action) + V_end[next_state(cell,
action)]` for each action and takes the log-sum-exp; the forward pass
scatters mass per the resulting probability tensor, applying the
state transitions.

This generalises `LogitChoice` (where the only state change is the
choice axis itself) and gates against the wealth-axis interpolation
that `WealthChange` provides (state transitions that change a
continuous axis need the same `convert_distribution!` machinery).

## Why this is delicate

The fused stage mixes:

- A **continuous-axis transition** (wealth, requiring interpolation
  and the `convert_distribution!` share-based push).
- A **categorical transition** (h, requiring an integer-policy
  scatter).

It's not just "do `LogitChoice` then `WealthChange`" because the
wealth change is per-action: the backward log-sum-exp has to evaluate
`V_end` at the *transitioned* state, which means interpolating along
wealth for the `sell` action while exact-indexing along h for both
actions. This is the operation `apply_wealth_change_V!` plus a logsumexp
across actions.

## Tests it would need

- Backward log-sum-exp gives the right `V_choosesell` on a simple
  2-cell example (one homeowner, one renter).
- Forward scatter conserves mass.
- Wealth change is applied *only* to sellers; non-sellers keep their
  wealth.
- Duality identity.
- Composition with the rest of the housing-model chain works.

## Recommendation when this is built

Build `ChoiceWithTransition` (option above) rather than a one-off
`SellHome`. The generality earns its keep: durable-good adjustment,
firm entry/exit-with-payoff, retirement-with-pension-payout, all
share the "choice + state transition per action" structure.
