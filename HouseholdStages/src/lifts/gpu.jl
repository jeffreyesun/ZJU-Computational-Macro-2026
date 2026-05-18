"""
Re-build `stage` with array-typed static fields converted to `CuArray`,
so subsequent `allocate(stage_gpu, T)` produces GPU buffers and
`backward!` / `forward!` dispatch through CUDA.jl on those buffers.

**Not yet implemented.** The path forward is to add `CUDA` as an
optional / extension dep to `HouseholdStages/Project.toml`, define a
per-stage method that rebuilds with `cu(field)` on array-typed static
fields (transitions, policies) and re-allocated buffers, and let
algorithm-divergent methods (sparse scatter, monotone search) dispatch
on the concrete array type. Tracked as a deferred follow-up; see
`_attic/CVIAYN_core_code/MIGRATION.md` Step 20 for the original
sequencing context.
"""
function lift_gpu(stage::AbstractStage)
    error("lift_gpu is not yet implemented. See `HouseholdStages/src/lifts/gpu.jl`.")
end
