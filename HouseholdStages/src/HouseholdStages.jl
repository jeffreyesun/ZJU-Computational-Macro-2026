module HouseholdStages

# Public surface for the household-layer package. The categorical
# content (stage composition, product, K-operator framing, per-stage
# functorial lifts, V/Λ duality, sequence-space utilities) lives here.
# Outer-loop machinery (EquilibriumProblem, blocks, train_vnet) was
# dropped in the 2026-05-13 refactor; the 2026-05-19 refactor split
# every stage into Spec/Buffer/Stage trichotomy.
#
# The 2026-05-20 follow-up extended the Spec/Buffer ↔ Stage dispatch
# convention from `backward!`/`forward!` to every outer-loop helper:
# `solve_vfi_steady_state_given_env!`,
# `solve_lambda_steady_state_given_env!`,
# `solve_steady_state_given_env!`,
# `solve_transition_given_env_path!`, and `compute_direct_jacobian!`
# each have a Spec/Buffer-keyed primitive (in `outer_loop_internal.jl`)
# and a Stage-keyed public wrapper (in `outer_loop.jl`). See
# REFACTOR_PLAN.md and HOUSEHOLD_STAGES_REFACTOR_PLAN.md at the repo
# root.

using ForwardDiff: ForwardDiff, Dual

export
    # Layout
    AxisKind, ContinuousGrid, DiscreteFinite,
    StateAxis, StateLayout,
    continuous_grid, discrete_finite, categorical,
    axis_position, axisname, axisnames, axissize, axisvalues,
    layout_size, cells, cell_array, drop_axis,
    # Stage-parameter env resolution
    resolve, FromEnv,
    # Interpolation helpers
    reinterpolate!, reinterpolate_arr!,
    convert_distribution!, convert_distribution_arr!,
    k1_argmax_monotone!, k1_argmax_dc!,
    # Stage interface
    AbstractStage, AbstractStageSpec, AbstractStageBuffer, StageBuffer,
    allocate, allocate_kernel, allocate_scratch,
    backward!, forward!,
    V_start_buffer, Λ_end_buffer, input_layout, output_layout,
    bundle, invalidate!, @definestage,
    default_eltype,
    # Stage dependency machinery
    static_env_deps, effective_env_slice, validate_env, chain_env_names,
    env_schema, make_env,
    # Concrete stages (bundled types only — Spec/Buffer are internal)
    MarkovStage,
    ArgmaxStage, LogitChoiceStage,
    MigrationStage,
    WealthChangeStage,
    AssetPriceChangeStage,
    ConsumptionSavingsStage,
    ForgetfulSumStage,
    IdentityStage,
    UtilityStage,
    BorrowingConstraintStage,
    # Composition (∘ and × are the canonical operators; ∘ₛ/×ₛ are
    # one-cycle deprecation aliases that share the same methods)
    ChainStage, ∘ₛ,
    # Product
    ProductStage, product, ×, ×ₛ, replicate_age,
    # Moments
    MomentSpec, at_end,
    define_moment!, define_moments!, compute_moments,
    # Aggregate-Jacobian utilities (sequence-space)
    expectation_vectors, build_F, J_from_F,
    # Lifts
    lift_jacobian, with_eltype,
    backward_adjoint!, forward_adjoint!,
    lift_gpu,
    # Outer-loop computation surface — Spec/Buffer-keyed primitives
    # (in outer_loop_internal.jl) and Stage-keyed public wrappers
    # (in outer_loop.jl) share these names; dispatch routes them.
    solve_vfi_steady_state_given_env!,
    solve_lambda_steady_state_given_env!,
    solve_steady_state_given_env!,
    solve_transition_given_env_path!,
    compute_direct_jacobian!

include("layout.jl")
include("stages/axis_ops.jl")
include("helper/interpolations.jl")
include("stages/abstract.jl")
include("stages/markov_along.jl")
include("stages/argmax.jl")
include("stages/logit_choice.jl")
include("stages/migration.jl")
include("stages/wealth_change.jl")
include("stages/asset_price_change.jl")
include("stages/consumption_savings.jl")
include("stages/forgetful_sum.jl")
include("stages/identity_stage.jl")
include("stages/utility.jl")
include("stages/borrowing_constraint.jl")
include("stages/composition.jl")
include("stages/product.jl")
include("moments.jl")
include("lifts/jacobian.jl")
include("lifts/gpu.jl")
include("sequence_space.jl")
include("outer_loop_internal.jl")
include("outer_loop.jl")

end # module HouseholdStages
