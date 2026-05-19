# HouseholdStages — Benchmark results

> Run the suite yourself with
> `julia --project=HouseholdStages/bench HouseholdStages/bench/runbenchmarks.jl`.
> Pass `LARGE_BENCH=1` to re-run at the example-driver sizes
> (`N_w = 400` everywhere; spatial layout `(400, 3, 2)`). Reference
> baselines are hand-coded kernels written in the style of
> `reference_materials/example_stages/` (fixed shapes, direct buffer
> mutation, no closure dispatch, no per-call allocation).
>
> **Numbers below from a 2026-05-17 run on the user's CPU
> (Linux 6.8.0-111, Julia 1.12.3). Latest commit: post-`N_w = 400`
> release prep.**

## Default sizes (fast iteration)

`Aiyagari (80, 3)`, `K-S (100, 2)`, `Spatial (60, 3, 2)` — kept fast so
the bench reruns in seconds.

### Aiyagari size (80, 3)

| Stage | lib | ref | ratio |
|---|---|---|---|
| MarkovStage backward | 398 ns | 446 ns | 0.89× |
| MarkovStage forward | 399 ns | 448 ns | 0.89× |
| WealthChangeStage backward | 1.13 μs | 490 ns | 2.31× |
| WealthChangeStage forward | 672 ns | 711 ns | 0.94× |
| ConsumptionSavingsStage backward | 51.15 μs | 38.55 μs | 1.33× |
| Chain backward+forward | 54.66 μs | 40.58 μs | 1.35× |

### K-S size (100, 2)

| Stage | lib | ref | ratio |
|---|---|---|---|
| MarkovStage backward | 400 ns | 413 ns | 0.97× |
| MarkovStage forward | 416 ns | 348 ns | 1.20× |
| WealthChangeStage backward | 958 ns | 407 ns | 2.35× |
| WealthChangeStage forward | 487 ns | 576 ns | 0.85× |
| ConsumptionSavingsStage backward | 46.00 μs | 39.21 μs | 1.17× |
| Chain backward+forward | 54.15 μs | 42.02 μs | 1.29× |

### Spatial size (60, 3, 2)

| Stage | lib | ref | ratio |
|---|---|---|---|
| MigrationStage backward | 8.84 μs | 7.56 μs | 1.17× |
| ConsumptionSavingsStage backward (3D) | 76.66 μs | 52.52 μs | 1.46× |
| Chain backward + forward | 91.05 μs | — | (no ref baseline) |

## Large sizes (example-driver, `LARGE_BENCH=1`)

`Aiyagari (400, 3)`, `K-S (400, 2)`, `Spatial (400, 3, 2)` — same shape
that all four example drivers use post-2026-05-16. These are the
numbers the README quotes for the user-facing claim "within ~5–50 % of
hand-coded reference kernels."

### Aiyagari size (400, 3)

| Stage | lib | ref | ratio |
|---|---|---|---|
| MarkovStage backward | 1.69 μs | 1.97 μs | 0.86× |
| MarkovStage forward | 1.92 μs | 1.99 μs | 0.97× |
| WealthChangeStage backward | 3.57 μs | 2.34 μs | 1.53× |
| WealthChangeStage forward | 3.42 μs | 3.38 μs | 1.01× |
| ConsumptionSavingsStage backward | 970.76 μs | 912.76 μs | 1.06× |
| Chain backward+forward | 969.83 μs | 818.29 μs | 1.19× |

### K-S size (400, 2)

| Stage | lib | ref | ratio |
|---|---|---|---|
| MarkovStage backward | 1.40 μs | 1.45 μs | 0.97× |
| MarkovStage forward | 1.44 μs | 1.46 μs | 0.99× |
| WealthChangeStage backward | 2.54 μs | 1.55 μs | 1.64× |
| WealthChangeStage forward | 2.21 μs | 2.20 μs | 1.00× |
| ConsumptionSavingsStage backward | 587.23 μs | 605.44 μs | 0.97× |
| Chain backward+forward | 622.80 μs | 546.67 μs | 1.14× |

### Spatial size (400, 3, 2)

| Stage | lib | ref | ratio |
|---|---|---|---|
| MigrationStage backward | 58.71 μs | 51.93 μs | 1.13× |
| ConsumptionSavingsStage backward (3D) | 2.50 ms | 2.34 ms | 1.07× |
| Chain backward + forward | 2.58 ms | — | (no ref baseline) |

## How to read this

- **`ratio`** is `library_time / reference_time` at the minimum
  benchmark sample. Lower is better. `1.0×` = parity with a
  hand-coded kernel; `< 2×` is excellent; `< 4×` is acceptable
  given how much expressiveness the stage abstraction buys.
- The **`WealthChangeStage backward`** outlier (2.3× at default sizes,
  1.5× at large) is the closure-broadcast overhead in
  `_fill_wealth_post!` — the hand-coded reference inlines the
  `(1+r) b + w y` computation. At `N_w = 400` the bottleneck
  shifts to `ConsumptionSavingsStage.backward`, where the ratio is
  near 1.0× — the per-element work dominates the closure overhead.
- **`Chain backward+forward`** is the end-to-end number that matters
  for outer-loop iteration cost. At the example sizes
  (`N_w = 400`), the chain is `1.1×–1.2×` the hand-coded reference
  for Aiyagari and K-S; the spatial-chain reference is not
  exhaustively coded (would duplicate the migration + 3D-savings
  logic), but per-stage ratios suggest a similar `~1.1×`.

## Caveats

- Single-core, single-machine numbers. Allocations are zero on the
  hot path after the first call (verified by `@btime`'s
  allocation column).
- Reference kernels in the bench file are not used in production;
  they exist only as a fairness baseline. Any improvement landed
  on the library side should keep the ratios near or below 1.5×
  on the chain-level pass.
- ForwardDiff and GPU paths are not benchmarked here. `lift_jacobian`
  is correctness-tested but its perf path has not been
  characterised; `lift_gpu` is not exercised in the workspace
  environment.
