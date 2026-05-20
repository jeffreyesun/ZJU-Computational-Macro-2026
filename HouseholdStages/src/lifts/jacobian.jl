###########################################
# lift_jacobian (forward-mode + adjoints) #
###########################################

# Forward-mode AD rebuilds each stage's stage-owned buffers (V_start,
# Λ_end, plus any per-call workspace allocated through `allocate`) with
# eltype `ForwardDiff.Dual{Tag, T, N}`; running the existing `backward!`
# / `forward!` methods on Dual-typed inputs propagates tangents alongside
# the primal values, and the chain rule falls out of composition.
#
# Static fields (transition matrix, user closures, `Param` for ε, …) keep
# their original eltype; the cross-eltype matmul (`Float64 * Dual → Dual`)
# goes through LinearAlgebra's generic `mul!` fallback. The user's
# closures see whatever eltype is put in `env` — typically Dual when
# taking Jacobians.
#
# Reverse-mode shares the entry point but takes a different code path
# (per-stage adjoint methods), implemented below.
#
# Spec/Buffer/Stage trichotomy. `with_eltype` is keyed on Spec —
# rebuilding the Spec under a new `element_type`, then bundling a fresh
# Buffer. The bundled-stage one-liner is sugar: `with_eltype(stage, T) =
# bundle(with_eltype(stage.spec, T))`.

"""
    with_eltype(spec_or_stage, T::Type) -> rebuilt

Return a new instance whose buffer eltype is `T`. The primary method
takes an `AbstractStageSpec` and returns a new Spec with
`element_type = T` and `Param` fields re-typed; the bundled-stage
sugar bundles a fresh buffer from that Spec. Each concrete Spec type
implements its own method; the generic fallback raises.

This is the workhorse for [`lift_jacobian`](@ref) (rebuild with
`T = ForwardDiff.Dual{…}`).
"""
function with_eltype(spec::AbstractStageSpec, ::Type{T}) where {T}
    error("with_eltype not implemented for $(typeof(spec)). " *
          "Add a method in src/lifts/jacobian.jl.")
end

# Bundled-stage sugar: rebuild Spec, bundle fresh buffer.
with_eltype(stage::AbstractStage, ::Type{T}) where {T} =
    bundle(with_eltype(stage.spec, T))

# Helper: re-type a `Param` so its leaf type matches the buffer eltype.
# Literal value `v::Number` gets `convert`-ed; Symbol mode is preserved.
function _retype_param(p::Param, ::Type{T}) where {T}
    v = p.val
    return v isa Symbol ? Param{T}(v) : Param{T}(convert(T, v))
end

# Helper: extract a Symbol from a `Val`-typed type parameter
# (e.g. `Val{:linear}` → `:linear`). Used by Specs that carry a
# `Val`-encoded option as a type parameter (WealthChange `extrap`,
# ConsumptionSavings `monotone_search`).
_val_to_sym(::Val{X}) where {X} = X
_val_type_to_sym(::Type{Val{X}}) where {X} = X

# Per-Spec `with_eltype` methods #
#--------------------------------#

function with_eltype(spec::MarkovStageSpec, ::Type{T}) where {T}
    return MarkovStageSpec(spec.input_layout;
                           axis         = spec.axis,
                           transition   = spec.transition,
                           element_type = T)
end

with_eltype(spec::IdentityStageSpec, ::Type{T}) where {T} =
    IdentityStageSpec(spec.input_layout; element_type = T)

function with_eltype(spec::UtilityStageSpec, ::Type{T}) where {T}
    return UtilityStageSpec(spec.input_layout;
                            utility      = spec.utility,
                            element_type = T)
end

function with_eltype(spec::ForgetfulSumStageSpec, ::Type{T}) where {T}
    return ForgetfulSumStageSpec(spec.input_layout;
                                 forget_axis  = spec.forget_axis,
                                 element_type = T)
end

function with_eltype(spec::ArgmaxStageSpec, ::Type{T}) where {T}
    return ArgmaxStageSpec(spec.input_layout;
                           choice_axis    = spec.choice_axis,
                           flow_payoff    = spec.flow_payoff,
                           next_state_idx = spec.next_state_idx,
                           element_type   = T)
end

function with_eltype(spec::LogitChoiceStageSpec, ::Type{T}) where {T}
    return LogitChoiceStageSpec(spec.input_layout;
                                choice_axis    = spec.choice_axis,
                                flow_payoff    = spec.flow_payoff,
                                next_state_idx = spec.next_state_idx,
                                ε              = _retype_param(spec.ε, T),
                                element_type   = T)
end

function with_eltype(spec::WealthChangeStageSpec{F,T0,L,Extrap},
                     ::Type{T}) where {F,T0,L,Extrap,T}
    return WealthChangeStageSpec(spec.input_layout;
                                 wealth_post  = spec.wealth_post,
                                 wealth_axis  = spec.wealth_axis,
                                 extrap       = _val_type_to_sym(Extrap),
                                 element_type = T)
end

function with_eltype(spec::ConsumptionSavingsStageSpec, ::Type{T}) where {T}
    return ConsumptionSavingsStageSpec(spec.input_layout;
                                       β               = _retype_param(spec.β, T),
                                       utility         = spec.utility,
                                       wealth_axis     = spec.wealth_axis,
                                       monotone_search = _val_to_sym(spec.monotone_search),
                                       element_type    = T)
end

function with_eltype(spec::BorrowingConstraintStageSpec, ::Type{T}) where {T}
    return BorrowingConstraintStageSpec(spec.input_layout;
                                        infeasible   = spec.infeasible,
                                        element_type = T)
end

function with_eltype(spec::MigrationStageSpec, ::Type{T}) where {T}
    return MigrationStageSpec(spec.input_layout;
                              location_axis  = spec.location_axis,
                              migration_cost = spec.migration_cost,
                              amenity        = _retype_migration_amenity(spec.amenity, T),
                              ε              = _retype_param(spec.ε, T),
                              element_type   = T)
end

# Helper: re-type the destination amenity field for `with_eltype`.
# `nothing` is unchanged; a static vector is converted elementwise; a
# closure has no natural element type and is passed through. Mirrors
# `_retype_param`'s policy of preserving symbolic / closure forms.
_retype_migration_amenity(::Nothing, ::Type) = nothing
_retype_migration_amenity(v::AbstractVector, ::Type{T}) where {T} =
    convert(AbstractVector{T}, v)
_retype_migration_amenity(f, ::Type) = f

# A re-typed chain preserves moment specs (they're static closures, not
# arrays); only the per-stage buffers change.
function with_eltype(spec::ChainStageSpec, ::Type{T}) where {T}
    new_stages = map(s -> with_eltype(s, T), spec.stages)
    return ChainStageSpec(new_stages; moments = spec.moments)
end

function with_eltype(spec::ProductStageSpec, ::Type{T}) where {T}
    new_components = map(s -> with_eltype(s, T), spec.components)
    return ProductStageSpec(new_components; axis = spec.axis, element_type = T)
end

# lift_jacobian #
#---------------#

"""
Forward-mode (`mode=:forward`): rebuild `stage` with
`ForwardDiff.Dual{tag, primal_eltype, n_dual}`-typed buffers, returning a
new stage / chain that can be `backward!` / `forward!`-ed on Dual-typed
inputs. The user typically wraps the rebuilt chain in
`ForwardDiff.derivative` or `ForwardDiff.jacobian`.

Reverse-mode (`mode=:reverse`) returns the stage unchanged; reverse-mode
is exposed via per-stage [`backward_adjoint!`](@ref) /
[`forward_adjoint!`](@ref) methods plus chain-level adjoint walks.
"""
function lift_jacobian(stage::AbstractStage;
                       mode::Symbol             = :forward,
                       n_dual::Int              = 1,
                       tag::Type                = Nothing,
                       primal_eltype::Type      = Float64)
    if mode === :forward
        dual_eltype = ForwardDiff.Dual{tag, primal_eltype, n_dual}
        return with_eltype(stage, dual_eltype)
    elseif mode === :reverse
        # The user gets back the stage unchanged; reverse-mode is exposed
        # via the per-stage `backward_adjoint!` / `forward_adjoint!`
        # methods plus chain-level adjoint walks (see below).
        return stage
    else
        error("lift_jacobian: unknown mode :$mode (expected :forward or :reverse)")
    end
end

##########################
# Reverse-mode adjoints #
##########################

# Conventions. Let `K` be a stage's K-operator (linear on measures /
# functions). The four operations of interest are:
#
#   forward (primal)            Λ_end   = K   * Λ_start
#   backward (primal)           V_start = K^T * V_end       (+ flow payoff r)
#   forward_adjoint (VJP)       dΛ_start = K^T * dΛ_end
#   backward_adjoint (VJP)      dV_end   = K   * dV_start
#
# For *linear-K* stages — those whose K is V/θ-independent (MarkovStage,
# IdentityStage, ForgetfulSumStage) — the adjoint operations are just the
# corresponding primal operations on the dual quantity, applied with
# fresh buffers to avoid clobbering the primal.
#
# Choice-stage adjoints (ArgmaxStage, LogitChoiceStage, ConsumptionSavingsStage, MigrationStage)
# reuse the linear K materialized at the primal eval-point: although K depends on
# V_out via the policy / probabilities, the policy is stored in the
# kernel after the primal pass, so the chain-rule VJP at a fixed
# eval-point reuses that linear K. The envelope theorem justifies
# ignoring the policy's V_out-dependence at interior cells; at boundary
# cells where two actions tie, the result is a subgradient.
#
# Spec/Buffer/Stage trichotomy. Adjoints are keyed on Spec, taking the
# matching Buffer as the third argument. Bundled-stage one-liners
# delegate to the Spec-keyed primary.

"""
    backward_adjoint!(spec, dV_start, buffer) -> dV_end
    backward_adjoint!(stage, dV_start)        -> dV_end

Given the sensitivity `dV_start = ∂L/∂V_start` of a downstream loss
w.r.t. `V_start`, return `dV_end = ∂L/∂V_end`. The relation `dV_end =
K · dV_start` holds for any linear-K stage; non-linear-K stages must
override. The Spec-keyed signature is the primary; the bundled-stage
form is one-line sugar.
"""
function backward_adjoint!(spec::AbstractStageSpec, dV_start, buffer)
    error("backward_adjoint! not implemented for $(typeof(spec)). " *
          "Linear-K stages (MarkovStage, IdentityStage, ForgetfulSumStage, " *
          "WealthChangeStage) ship default adjoints; choice-stage adjoints " *
          "(ArgmaxStage, LogitChoiceStage, ConsumptionSavingsStage, MigrationStage) use the " *
          "kernel's materialised K (envelope theorem) and have explicit methods.")
end

backward_adjoint!(stage::AbstractStage, dV_start) =
    backward_adjoint!(stage.spec, dV_start, stage.buffer)

"""
    forward_adjoint!(spec, dΛ_end, buffer) -> dΛ_start
    forward_adjoint!(stage, dΛ_end)        -> dΛ_start

Given `dΛ_end = ∂L/∂Λ_end`, return `dΛ_start = ∂L/∂Λ_start`. For
linear-K stages `dΛ_start = K^T · dΛ_end`; non-linear-K stages must
override. The Spec-keyed signature is the primary; the bundled-stage
form is one-line sugar.
"""
function forward_adjoint!(spec::AbstractStageSpec, dΛ_end, buffer)
    error("forward_adjoint! not implemented for $(typeof(spec)). " *
          "Same caveats as backward_adjoint!.")
end

forward_adjoint!(stage::AbstractStage, dΛ_end) =
    forward_adjoint!(stage.spec, dΛ_end, stage.buffer)

# MarkovStage #
#-------------#
# K = transition'; forward applies K (i.e., transition'); backward
# applies K^T = transition. The adjoint of forward is the application
# of K^T to a sensitivity, which is exactly the backward-style apply.

function backward_adjoint!(spec::MarkovStageSpec, dV_start,
                           buffer::MarkovStageBuffer)
    scratch = buffer.scratch
    dV_end = similar(dV_start)
    perm_in  = similar(scratch.perm_in)
    perm_out = similar(scratch.perm_out)
    # K · dV_start = transition' · dV_start along axis_dim.
    _markov_apply!(dV_end, dV_start, spec.transition',
                   spec.axis_dim, perm_in, perm_out)
    return dV_end
end

function forward_adjoint!(spec::MarkovStageSpec, dΛ_end,
                          buffer::MarkovStageBuffer)
    scratch = buffer.scratch
    dΛ_start = similar(dΛ_end)
    perm_in  = similar(scratch.perm_in)
    perm_out = similar(scratch.perm_out)
    # K^T · dΛ_end = transition · dΛ_end along axis_dim.
    _markov_apply!(dΛ_start, dΛ_end, spec.transition,
                   spec.axis_dim, perm_in, perm_out)
    return dΛ_start
end

# IdentityStage #
#---------------#
# K = I. Both adjoints are the identity.

backward_adjoint!(::IdentityStageSpec, dV_start::AbstractArray,
                  ::IdentityStageBuffer) = copy(dV_start)
forward_adjoint!(::IdentityStageSpec, dΛ_end::AbstractArray,
                 ::IdentityStageBuffer) = copy(dΛ_end)

# ForgetfulSumStage #
#-------------------#
# K: sum-along-axis. dim_in has one extra axis vs dim_out. Forward sums
# along the dropped axis; backward broadcasts (K^T applied to V_end is
# V_start of larger shape, constant along the dropped axis).
# Adjoint of forward: dΛ_start = K^T · dΛ_end (broadcast).
# Adjoint of backward: dV_end = K · dV_start (sum).

function forward_adjoint!(spec::ForgetfulSumStageSpec{T,LIn,LOut},
                          dΛ_end,
                          ::ForgetfulSumStageBuffer) where {T,LIn,LOut}
    dims_out = layout_size(spec.output_layout)
    @assert size(dΛ_end) == dims_out
    dims_in = layout_size(spec.input_layout)
    Nin     = length(dims_in)
    dΛ_start = Array{T, Nin}(undef, dims_in)
    shape    = _insert_singleton(dims_out, spec.forget_dim)
    dΛ_start .= reshape(dΛ_end, shape)
    return dΛ_start
end

function backward_adjoint!(spec::ForgetfulSumStageSpec{T,LIn,LOut},
                           dV_start,
                           ::ForgetfulSumStageBuffer) where {T,LIn,LOut}
    dims_in  = layout_size(spec.input_layout)
    dims_out = layout_size(spec.output_layout)
    Nout     = length(dims_out)
    @assert size(dV_start) == dims_in
    dV_end = Array{T, Nout}(undef, dims_out)
    shape  = _insert_singleton(dims_out, spec.forget_dim)
    sum!(reshape(dV_end, shape), dV_start)
    return dV_end
end

# ChainStage #
#------------#
# Reverse walks: backward_adjoint walks stages in forward order (since
# backward primal walks reverse, adjoint of that walks forward).
# forward_adjoint walks in reverse. Both read from
# `spec.stages[i]` and `buffer.stages[i]` in lockstep.

function backward_adjoint!(spec::ChainStageSpec, dV_start,
                           buffer::ChainStageBuffer)
    dV = dV_start
    for i in eachindex(spec.stages)
        dV = backward_adjoint!(spec.stages[i], dV, buffer.stages[i])
    end
    return dV
end

function forward_adjoint!(spec::ChainStageSpec, dΛ_end,
                          buffer::ChainStageBuffer)
    dΛ = dΛ_end
    n  = length(spec.stages)
    for i in n:-1:1
        dΛ = forward_adjoint!(spec.stages[i], dΛ, buffer.stages[i])
    end
    return dΛ
end

# ArgmaxStage #
#-------------#

function backward_adjoint!(spec::ArgmaxStageSpec, dV_in,
                           buffer::ArgmaxStageBuffer)
    # Primal:  V_in[s]  = r(s, π(s)) + V_out[ν(s, π(s))]
    # Adjoint: dV_out[c'] = Σ_{s : ν(s, π(s)) = c'} dV_in[s]
    #                    = K · dV_in    (scatter via the policy)
    (; input_layout, choice_dim) = spec
    policy  = buffer.kernel.policy
    actions = axisvalues(input_layout.axes[choice_dim])
    T = eltype(dV_in)
    dV_out  = zeros(T, size(dV_in))

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        action  = actions[policy[ci_in]]
        next_axis_i = spec.next_state_idx(cell, action)
        out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
        dV_out[CartesianIndex(out_idxs)] += dV_in[ci_in]
    end
    return dV_out
end

function forward_adjoint!(spec::ArgmaxStageSpec, dΛ_end,
                          buffer::ArgmaxStageBuffer)
    # Primal:  Λ_end[c'] = Σ_{s : ν(s, π(s)) = c'} Λ_start[s]
    # Adjoint: dΛ_start[s] = dΛ_end[ν(s, π(s))]    (gather via the policy)
    (; input_layout, choice_dim) = spec
    policy  = buffer.kernel.policy
    actions = axisvalues(input_layout.axes[choice_dim])
    dΛ_start = similar(dΛ_end)

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        action  = actions[policy[ci_in]]
        next_axis_i = spec.next_state_idx(cell, action)
        out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
        dΛ_start[ci_in] = dΛ_end[CartesianIndex(out_idxs)]
    end
    return dΛ_start
end

# LogitChoiceStage #
#------------------#

function backward_adjoint!(spec::LogitChoiceStageSpec, dV_in,
                           buffer::LogitChoiceStageBuffer)
    # Primal:  V_in[s] = ε log Σ_a exp((r + V_out[ν(s,a)]) / ε)
    #          ∂V_in/∂V_out[c'] = Σ_a P(a|s) · I(c' = ν(s,a))
    # Adjoint: dV_out[c'] = Σ_{s, a} dV_in[s] · P(a|s) · I(c' = ν(s,a))
    (; input_layout, choice_dim) = spec
    actions = axisvalues(input_layout.axes[choice_dim])
    prob    = buffer.kernel.choice_prob
    T       = eltype(dV_in)
    dV_out  = zeros(T, size(dV_in))

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        d       = dV_in[ci_in]
        iszero(d) && continue
        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = spec.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            dV_out[CartesianIndex(out_idxs)] += d * p
        end
    end
    return dV_out
end

function forward_adjoint!(spec::LogitChoiceStageSpec, dΛ_end,
                          buffer::LogitChoiceStageBuffer)
    # Primal:  Λ_end[c'] = Σ_{s, a} P(a|s) · Λ_start[s] · I(c' = ν(s,a))
    # Adjoint: dΛ_start[s] = Σ_a P(a|s) · dΛ_end[ν(s,a)]
    (; input_layout, choice_dim) = spec
    actions = axisvalues(input_layout.axes[choice_dim])
    prob    = buffer.kernel.choice_prob
    T       = eltype(dΛ_end)
    dΛ_start = zeros(T, size(dΛ_end))

    for (idx, cell) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        acc = zero(T)
        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = spec.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, choice_dim)
            acc += p * dΛ_end[CartesianIndex(out_idxs)]
        end
        dΛ_start[ci_in] = acc
    end
    return dΛ_start
end

# MigrationStage #
#----------------#
# Same structure as LogitChoiceStage, but the destination is the action index
# directly (no `next_state_idx` indirection).

"""CLAUDE
Reverse-mode backward-adjoint for [`MigrationStage`](@ref): mirror of the
LogitChoiceStage adjoint, with destination = action-index identity.
"""
function backward_adjoint!(spec::MigrationStageSpec, dV_in,
                           buffer::MigrationStageBuffer)
    (; input_layout, location_dim) = spec
    n_loc  = axissize(input_layout.axes[location_dim])
    prob   = buffer.kernel.choice_prob
    T      = eltype(dV_in)
    dV_out = zeros(T, size(dV_in))
    dims   = layout_size(input_layout)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        d = dV_in[ci]
        iszero(d) && continue
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, location_dim)
            dV_out[CartesianIndex(out_idxs)] += d * p
        end
    end
    return dV_out
end

"""CLAUDE
Reverse-mode forward-adjoint for [`MigrationStage`](@ref): mirror of the
LogitChoiceStage adjoint.
"""
function forward_adjoint!(spec::MigrationStageSpec, dΛ_end,
                          buffer::MigrationStageBuffer)
    (; input_layout, location_dim) = spec
    n_loc    = axissize(input_layout.axes[location_dim])
    prob     = buffer.kernel.choice_prob
    T        = eltype(dΛ_end)
    dΛ_start = zeros(T, size(dΛ_end))
    dims     = layout_size(input_layout)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        acc = zero(T)
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, location_dim)
            acc += p * dΛ_end[CartesianIndex(out_idxs)]
        end
        dΛ_start[ci] = acc
    end
    return dΛ_start
end

# ConsumptionSavingsStage #
#-------------------------#
# Structure identical to ArgmaxStage — a sparse-permutation kernel where the
# "policy" is the next-wealth grid index per input cell. Implemented
# for SSJ's `expectation_vectors` on the 3-stage chain
# `IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavingsStage`.

"""CLAUDE
Reverse-mode forward-adjoint for `ConsumptionSavingsStage`: K is the
sparse permutation defined by the policy, so `K^T · dΛ_end` is a
gather along the wealth axis at each cell's stored policy index.
"""
function forward_adjoint!(spec::ConsumptionSavingsStageSpec, dΛ_end,
                          buffer::ConsumptionSavingsStageBuffer)
    (; input_layout, wealth_dim) = spec
    policy   = buffer.kernel.policy
    dΛ_start = similar(dΛ_end)

    for (idx, _) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        a1_i    = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a1_i, wealth_dim)
        dΛ_start[ci_in] = dΛ_end[CartesianIndex(out_idxs)]
    end
    return dΛ_start
end

"""CLAUDE
Reverse-mode backward-adjoint for `ConsumptionSavingsStage`: dual of the
above gather is a scatter `dV_out[c'] += dV_in[s]` along the policy.
"""
function backward_adjoint!(spec::ConsumptionSavingsStageSpec, dV_in,
                           buffer::ConsumptionSavingsStageBuffer)
    (; input_layout, wealth_dim) = spec
    policy = buffer.kernel.policy
    T      = eltype(dV_in)
    dV_out = zeros(T, size(dV_in))

    for (idx, _) in cells(input_layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        a1_i    = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a1_i, wealth_dim)
        dV_out[CartesianIndex(out_idxs)] += dV_in[ci_in]
    end
    return dV_out
end

# WealthChangeStage #
#-------------------#
# Primal forward: K is the share-based linear redistribution of mass at
# per-cell source positions `wpost` onto the canonical wgrid. With
# underflow / overflow accumulating at the endpoints (matching
# `convert_distribution!`'s policy in
# `src/helper/interpolations.jl`), the K-transpose at a source cell `s`
# is the share-weighted gather from `wgrid`'s two neighbors of
# `wpost[s]` — i.e., linear interpolation at `wpost[s]` against the
# input expectation array on `wgrid`, with clip-on-both-ends extrap.
#
# This is NOT identical to `reinterpolate!` with `:clip`, because that
# helper clips on the left but linearly extrapolates on the right.
# `_share_gather!` below implements the exact dual of
# `convert_distribution!`.

"""CLAUDE
K-transpose gather for [`WealthChangeStage`](@ref): apply linear
interpolation of `dΛ_end` (1-D vector on `wgrid`) at each source
position in `wpost_slice`, writing into `dΛ_start_slice`. Underflow
clips to `dΛ_end[1]`; overflow clips to `dΛ_end[end]` — matching the
endpoint accumulation rule of [`convert_distribution!`](@ref).
"""
function _share_gather!(dΛ_start_slice::AbstractVector{T},
                        dΛ_end::AbstractVector{T},
                        wpost_slice::AbstractVector{T},
                        wgrid::AbstractVector{T}) where {T}
    n_w  = length(wgrid)
    w_lo = wgrid[1]
    w_hi = wgrid[end]
    j = 1
    @inbounds for i in eachindex(dΛ_start_slice)
        wi = wpost_slice[i]
        if wi < w_lo
            dΛ_start_slice[i] = dΛ_end[1]
        elseif wi >= w_hi
            dΛ_start_slice[i] = dΛ_end[end]
        else
            while wi >= wgrid[j+1]
                j += 1
                j == n_w - 1 && break
            end
            # Walk j back if wpost is non-monotone across cells.
            while j > 1 && wi < wgrid[j]
                j -= 1
            end
            left_share = (wgrid[j+1] - wi) / (wgrid[j+1] - wgrid[j])
            dΛ_start_slice[i] =
                left_share * dΛ_end[j] + (1 - left_share) * dΛ_end[j+1]
        end
    end
    return dΛ_start_slice
end

"""CLAUDE
Reverse-mode forward-adjoint for [`WealthChangeStage`](@ref): K-transpose
applied to `dΛ_end` is a per-source-cell gather from `wgrid` via
share-based linear interpolation at the materialised `wpost` value.

Iterates over slices along the wealth axis (the kernel stores `wpost`
as a per-cell N-D array; the other axes are pass-through).
"""
function forward_adjoint!(spec::WealthChangeStageSpec, dΛ_end,
                          buffer::WealthChangeStageBuffer)
    (; input_layout, wealth_dim) = spec
    T     = eltype(dΛ_end)
    wgrid = collect(T, axisvalues(input_layout.axes[wealth_dim]))
    wpost = buffer.kernel.wealth_post
    N     = ndims(dΛ_end)

    dΛ_start = similar(dΛ_end)
    dims  = size(dΛ_end)
    other = ntuple(i -> i == wealth_dim ? 1 : dims[i], N)

    if wealth_dim == 1
        for other_ci in CartesianIndices(other)
            tail = other_ci.I[2:end]
            dΛ_start_view = view(dΛ_start, :, tail...)
            dΛ_end_view   = view(dΛ_end,   :, tail...)
            wpost_view    = view(wpost,    :, tail...)
            _share_gather!(dΛ_start_view, dΛ_end_view, wpost_view, wgrid)
        end
    else
        # General permutation: same approach as in `_along_wealth`.
        perm     = _bring_dim_first(N, wealth_dim)
        inv_perm = invperm(perm)
        dΛ_end_p   = permutedims(dΛ_end, perm)
        wpost_p    = permutedims(wpost,  perm)
        dΛ_start_p = similar(dΛ_end_p)
        permdims = ntuple(i -> dims[perm[i]], N)
        other_p  = ntuple(i -> i == 1 ? 1 : permdims[i], N)
        for other_ci in CartesianIndices(other_p)
            tail = other_ci.I[2:end]
            dΛ_start_view = view(dΛ_start_p, :, tail...)
            dΛ_end_view   = view(dΛ_end_p,   :, tail...)
            wpost_view    = view(wpost_p,    :, tail...)
            _share_gather!(dΛ_start_view, dΛ_end_view, wpost_view, wgrid)
        end
        dΛ_start .= permutedims(dΛ_start_p, inv_perm)
    end
    return dΛ_start
end

"""CLAUDE
Reverse-mode backward-adjoint for [`WealthChangeStage`](@ref): not yet
implemented. The primal `backward!` interpolates `V_end` linearly at
`wpost(cell)`, so the dual would be a share-based scatter — exactly
the structure of `convert_distribution!`. SSJ's `expectation_vectors`
pipeline does not need this method (it only uses `forward_adjoint!`),
so it is deliberately stubbed for now. If reverse-mode gradients
through the value function are needed later, mirror the `_share_gather!`
helper with a scatter pattern.
"""
function backward_adjoint!(spec::WealthChangeStageSpec, dV_start,
                           buffer::WealthChangeStageBuffer)
    error("WealthChangeStage.backward_adjoint! not yet implemented. " *
          "Only `forward_adjoint!` (used by `expectation_vectors`) is " *
          "currently wired up; the backward adjoint would mirror the " *
          "share-based gather as a scatter — add when reverse-mode " *
          "gradients through V are needed.")
end
