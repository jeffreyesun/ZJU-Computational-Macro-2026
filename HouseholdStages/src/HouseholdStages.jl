module HouseholdStages

# Public surface for the household-layer package. The categorical
# content (stage composition, product, K-operator framing, per-stage
# functorial lifts, V/Λ duality, sequence-space utilities) lives here.
# Outer-loop machinery (EquilibriumProblem, blocks, train_vnet) was
# dropped in the 2026-05-13 refactor — see REFACTOR_PLAN.md and the
# attic at _attic/CVIAYN_core_code/ for that history.

using ForwardDiff: ForwardDiff, Dual

export
    # Layout
    AxisKind, ContinuousGrid, DiscreteFinite,
    StateAxis, StateLayout,
    continuous_grid, discrete_finite, categorical,
    axis_position, axisname, axisnames, axissize, axisvalues,
    layout_size, cells, cell_array, drop_axis,
    # Param wrapper
    Param, resolve, is_swept, swept_key,
    # Interpolation helpers
    reinterpolate!, reinterpolate_arr!,
    convert_distribution!, convert_distribution_arr!,
    k1_argmax_monotone!, k1_argmax_dc!,
    # Stage interface
    AbstractStage, allocate, backward!, forward!,
    V_start_buffer, Λ_end_buffer,
    # Stage dependency machinery
    static_env_deps, effective_env_slice, validate_env, chain_env_names,
    # Concrete stages
    MarkovAlong,
    Argmax, LogitChoice,
    Migration,
    WealthChange,
    AssetPriceChange,
    ConsumptionSavings,
    ForgetfulSum,
    IdentityStage,
    UtilityStage,
    BorrowingConstraint,
    # Composition
    StageChain, ∘ₛ,
    # Product
    ProductStage, product, ×ₛ, replicate_age,
    # Moments
    MomentSpec, at_end, lift_moments, MomentedChain, compute_moments,
    # Aggregate-Jacobian utilities (sequence-space)
    expectation_vectors, build_F, J_from_F,
    # Lifts
    lift_jacobian, with_eltype,
    backward_adjoint!, forward_adjoint!,
    lift_gpu,
    # Inner-solve helpers at a fixed env (2026-05-18, trimmed)
    solve_vfi_steady_state_given_env!,
    solve_lambda_steady_state_given_env!,
    solve_steady_state_given_env!

include("layout.jl")
include("param.jl")
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
include("outer_loop.jl")

end # module HouseholdStages
