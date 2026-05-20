"""
Re-build `spec` (or its bundled `stage`) with array-typed static fields
converted to `CuArray`, so subsequent `allocate(spec_gpu, T)` produces
GPU buffers and `backward!` / `forward!` dispatch through CUDA.jl on
those buffers.

**Not yet implemented.** The path forward is to add `CUDA` as an
optional / extension dep to `HouseholdStages/Project.toml`, define a
per-stage Spec method that rebuilds with `cu(field)` on array-typed
static fields (transitions, costs) and re-allocated buffers, and let
algorithm-divergent methods (sparse scatter, monotone search) dispatch
on the concrete array type. Tracked as a deferred follow-up; see
`_attic/CVIAYN_core_code/MIGRATION.md` Step 20 for the original
sequencing context.

The Spec-keyed signature is the primary; the bundled-stage form is
one-line sugar — `lift_gpu(stage) = bundle(lift_gpu(stage.spec))`.
"""
function lift_gpu(spec::AbstractStageSpec)
    error("lift_gpu is not yet implemented. See `HouseholdStages/src/lifts/gpu.jl`.")
end

lift_gpu(stage::AbstractStage) = bundle(lift_gpu(stage.spec))
