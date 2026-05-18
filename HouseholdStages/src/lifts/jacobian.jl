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

"""
    with_eltype(stage_or_chain, T::Type) -> rebuilt

Return a new instance of `stage_or_chain` whose buffer eltype is `T`.
Static fields (transition matrices, user closures, layouts) are
shared, not copied — only the workspace buffers are re-allocated. Each
concrete stage type implements its own method; the generic fallback
raises.

This is the workhorse for [`lift_jacobian`](@ref) (rebuild with
`T = ForwardDiff.Dual{…}`).
"""
function with_eltype(stage::AbstractStage, ::Type{T}) where {T}
    error("with_eltype not implemented for $(typeof(stage)). " *
          "Add a method in src/lifts/jacobian.jl.")
end

# Helper: re-type a `Param` so its leaf type matches the buffer eltype.
# Literal value `v::Number` gets `convert`-ed; Symbol mode is preserved.
function _retype_param(p::Param, ::Type{T}) where {T}
    v = p.val
    return v isa Symbol ? Param{T}(v) : Param{T}(convert(T, v))
end

# Per-stage `with_eltype` methods #
#---------------------------------#

function with_eltype(s::MarkovAlong, ::Type{T}) where {T}
    return MarkovAlong(s.input_layout;
                       axis         = s.axis,
                       transition   = s.transition,
                       element_type = T)
end

with_eltype(s::IdentityStage, ::Type{T}) where {T} =
    IdentityStage(s.input_layout; element_type = T)

function with_eltype(s::UtilityStage, ::Type{T}) where {T}
    return UtilityStage(s.input_layout;
                        utility      = s.utility,
                        closure_deps = s.closure_deps,
                        element_type = T)
end

function with_eltype(s::ForgetfulSum, ::Type{T}) where {T}
    return ForgetfulSum(s.input_layout;
                        forget_axis  = s.forget_axis,
                        element_type = T)
end

function with_eltype(s::Argmax, ::Type{T}) where {T}
    return Argmax(s.input_layout;
                  choice_axis    = s.choice_axis,
                  flow_payoff    = s.flow_payoff,
                  next_state_idx = s.next_state_idx,
                  closure_deps   = s.closure_deps,
                  element_type   = T)
end

function with_eltype(s::LogitChoice, ::Type{T}) where {T}
    return LogitChoice(s.input_layout;
                       choice_axis    = s.choice_axis,
                       flow_payoff    = s.flow_payoff,
                       next_state_idx = s.next_state_idx,
                       ε              = _retype_param(s.ε, T),
                       closure_deps   = s.closure_deps,
                       element_type   = T)
end

# Extract the symbol from `WealthChange`'s Extrap type-parameter
# (`Val{:linear}` etc.) for round-trip with_eltype.
_extrap_symbol(::Type{Val{X}}) where {X} = X

function with_eltype(s::WealthChange{F,T0,N,D,L,AV,Extrap}, ::Type{T}) where {F,T0,N,D,L,AV,Extrap,T}
    return WealthChange(s.input_layout;
                        wealth_post  = s.wealth_post,
                        wealth_axis  = s.wealth_axis,
                        closure_deps = s.closure_deps,
                        extrap       = _extrap_symbol(Extrap),
                        element_type = T)
end

_val_to_sym(::Val{X}) where {X} = X

function with_eltype(s::ConsumptionSavings, ::Type{T}) where {T}
    return ConsumptionSavings(s.input_layout;
                              β               = _retype_param(s.β, T),
                              utility         = s.utility,
                              wealth_axis     = s.wealth_axis,
                              closure_deps    = s.closure_deps,
                              monotone_search = _val_to_sym(s.monotone_search),
                              element_type    = T)
end

function with_eltype(s::BorrowingConstraint, ::Type{T}) where {T}
    return BorrowingConstraint(s.input_layout;
                               infeasible   = s.infeasible,
                               closure_deps = s.closure_deps,
                               element_type = T)
end

function with_eltype(s::Migration, ::Type{T}) where {T}
    return Migration(s.input_layout;
                     location_axis  = s.location_axis,
                     migration_cost = s.migration_cost,
                     ε              = _retype_param(s.ε, T),
                     closure_deps   = s.closure_deps,
                     element_type   = T)
end

with_eltype(c::StageChain, ::Type{T}) where {T} =
    StageChain(map(s -> with_eltype(s, T), c.stages))

with_eltype(m::MomentedChain, ::Type{T}) where {T} =
    MomentedChain(with_eltype(m.inner, T), m.specs, m.out_layout)

function with_eltype(p::ProductStage, ::Type{T}) where {T}
    new_components = map(s -> with_eltype(s, T), p.components)
    return product(new_components...; axis = p.axis)
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
# For *linear-K* stages — those whose K is V/θ-independent (MarkovAlong,
# IdentityStage, ForgetfulSum) — the adjoint operations are just the
# corresponding primal operations on the dual quantity, applied with
# fresh buffers to avoid clobbering the primal.
#
# Choice-stage adjoints (Argmax, LogitChoice, ConsumptionSavings, Migration)
# reuse the linear K materialized at the primal eval-point: although K depends on
# V_out via the policy / probabilities, the policy is stored in the
# kernel after the primal pass, so the chain-rule VJP at a fixed
# eval-point reuses that linear K. The envelope theorem justifies
# ignoring the policy's V_out-dependence at interior cells; at boundary
# cells where two actions tie, the result is a subgradient.

"""
    backward_adjoint!(stage, dV_start, kernel, scratch) -> dV_end

Given the sensitivity `dV_start = ∂L/∂V_start` of a downstream loss
w.r.t. `V_start`, return `dV_end = ∂L/∂V_end`. The relation `dV_end =
K · dV_start` holds for any linear-K stage; non-linear-K stages must
override.
"""
function backward_adjoint!(stage::AbstractStage, dV_start, kernel, scratch)
    error("backward_adjoint! not implemented for $(typeof(stage)). " *
          "Linear-K stages (MarkovAlong, IdentityStage, ForgetfulSum, " *
          "WealthChange) ship default adjoints; choice-stage adjoints " *
          "(Argmax, LogitChoice, ConsumptionSavings, Migration) use the " *
          "kernel's materialised K (envelope theorem) and have explicit methods.")
end

"""
    forward_adjoint!(stage, dΛ_end, kernel, scratch) -> dΛ_start

Given `dΛ_end = ∂L/∂Λ_end`, return `dΛ_start = ∂L/∂Λ_start`. For
linear-K stages `dΛ_start = K^T · dΛ_end`; non-linear-K stages must
override.
"""
function forward_adjoint!(stage::AbstractStage, dΛ_end, kernel, scratch)
    error("forward_adjoint! not implemented for $(typeof(stage)). " *
          "Same caveats as backward_adjoint!.")
end

# MarkovAlong #
#-------------#
# K = transition'; forward applies K (i.e., transition'); backward
# applies K^T = transition. The adjoint of forward is the application
# of K^T to a sensitivity, which is exactly the backward-style apply.

function backward_adjoint!(s::MarkovAlong{M,T,N},
                           dV_start::AbstractArray{T,N},
                           kernel, scratch) where {M,T,N}
    dV_end = similar(dV_start)
    perm_in  = similar(scratch.perm_in)
    perm_out = similar(scratch.perm_out)
    # K · dV_start = transition' · dV_start along axis_dim.
    _markov_apply!(dV_end, dV_start, s.transition',
                   s.axis_dim, perm_in, perm_out)
    return dV_end
end

function forward_adjoint!(s::MarkovAlong{M,T,N},
                          dΛ_end::AbstractArray{T,N},
                          kernel, scratch) where {M,T,N}
    dΛ_start = similar(dΛ_end)
    perm_in  = similar(scratch.perm_in)
    perm_out = similar(scratch.perm_out)
    # K^T · dΛ_end = transition · dΛ_end along axis_dim.
    _markov_apply!(dΛ_start, dΛ_end, s.transition,
                   s.axis_dim, perm_in, perm_out)
    return dΛ_start
end

# IdentityStage #
#---------------#
# K = I. Both adjoints are the identity.

backward_adjoint!(::IdentityStage, dV_start::AbstractArray, kernel, scratch) =
    copy(dV_start)
forward_adjoint!(::IdentityStage, dΛ_end::AbstractArray, kernel, scratch) =
    copy(dΛ_end)

# ForgetfulSum #
#--------------#
# K: sum-along-axis. dim_in has one extra axis vs dim_out. Forward sums
# along the dropped axis; backward broadcasts (K^T applied to V_end is
# V_start of larger shape, constant along the dropped axis).
# Adjoint of forward: dΛ_start = K^T · dΛ_end (broadcast).
# Adjoint of backward: dV_end = K · dV_start (sum).

function forward_adjoint!(s::ForgetfulSum{T,Nin,Nout},
                          dΛ_end::AbstractArray{T,Nout},
                          kernel, scratch) where {T,Nin,Nout}
    dims_out = layout_size(s.output_layout)
    @assert size(dΛ_end) == dims_out
    dims_in = layout_size(s.input_layout)
    dΛ_start = Array{T, Nin}(undef, dims_in)
    shape    = _insert_singleton(dims_out, s.forget_dim)
    dΛ_start .= reshape(dΛ_end, shape)
    return dΛ_start
end

function backward_adjoint!(s::ForgetfulSum{T,Nin,Nout},
                           dV_start::AbstractArray{T,Nin},
                           kernel, scratch) where {T,Nin,Nout}
    dims_in  = layout_size(s.input_layout)
    dims_out = layout_size(s.output_layout)
    @assert size(dV_start) == dims_in
    dV_end = Array{T, Nout}(undef, dims_out)
    shape    = _insert_singleton(dims_out, s.forget_dim)
    sum!(reshape(dV_end, shape), dV_start)
    return dV_end
end

# StageChain #
#------------#
# Reverse walks: backward_adjoint walks stages in forward order (since
# backward primal walks reverse, adjoint of that walks forward).
# forward_adjoint walks in reverse.

function backward_adjoint!(c::StageChain, dV_start, kernels::Tuple, scratches::Tuple)
    dV = dV_start
    for i in eachindex(c.stages)
        dV = backward_adjoint!(c.stages[i], dV, kernels[i], scratches[i])
    end
    return dV
end

function forward_adjoint!(c::StageChain, dΛ_end, kernels::Tuple, scratches::Tuple)
    dΛ = dΛ_end
    n  = length(c.stages)
    for i in n:-1:1
        dΛ = forward_adjoint!(c.stages[i], dΛ, kernels[i], scratches[i])
    end
    return dΛ
end

# MomentedChain delegates to the inner chain.
backward_adjoint!(m::MomentedChain, dV_start, kernels, scratches) =
    backward_adjoint!(m.inner, dV_start, kernels, scratches)
forward_adjoint!(m::MomentedChain, dΛ_end, kernels, scratches) =
    forward_adjoint!(m.inner, dΛ_end, kernels, scratches)

# Argmax #
#--------#

function backward_adjoint!(s::Argmax{F,BF,LIn,LOut,N,D,T},
                           dV_in::AbstractArray{T,N},
                           kernel, scratch) where {F,BF,LIn,LOut,N,D,T}
    # Primal:  V_in[s]  = r(s, π(s)) + V_out[ν(s, π(s))]
    # Adjoint: dV_out[c'] = Σ_{s : ν(s, π(s)) = c'} dV_in[s]
    #                    = K · dV_in    (scatter via the policy)
    layout = s.input_layout
    cdim   = s.choice_dim
    actions = axisvalues(layout.axes[cdim])
    policy  = s.policy
    dV_out  = zeros(T, size(dV_in))

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        action  = actions[policy[ci_in]]
        next_axis_i = s.next_state_idx(cell, action)
        out_idxs    = Base.setindex(in_idxs, next_axis_i, cdim)
        dV_out[CartesianIndex(out_idxs)] += dV_in[ci_in]
    end
    return dV_out
end

function forward_adjoint!(s::Argmax{F,BF,LIn,LOut,N,D,T},
                          dΛ_end::AbstractArray{T,N},
                          kernel, scratch) where {F,BF,LIn,LOut,N,D,T}
    # Primal:  Λ_end[c'] = Σ_{s : ν(s, π(s)) = c'} Λ_start[s]
    # Adjoint: dΛ_start[s] = dΛ_end[ν(s, π(s))]    (gather via the policy)
    layout = s.input_layout
    cdim   = s.choice_dim
    actions = axisvalues(layout.axes[cdim])
    policy  = s.policy
    dΛ_start = similar(dΛ_end)

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        action  = actions[policy[ci_in]]
        next_axis_i = s.next_state_idx(cell, action)
        out_idxs    = Base.setindex(in_idxs, next_axis_i, cdim)
        dΛ_start[ci_in] = dΛ_end[CartesianIndex(out_idxs)]
    end
    return dΛ_start
end

# LogitChoice #
#-------------#

function backward_adjoint!(s::LogitChoice{F,BF,LIn,LOut,N,D,T},
                           dV_in::AbstractArray{T,N},
                           kernel, scratch) where {F,BF,LIn,LOut,N,D,T}
    # Primal:  V_in[s] = ε log Σ_a exp((r + V_out[ν(s,a)]) / ε)
    #          ∂V_in/∂V_out[c'] = Σ_a P(a|s) · I(c' = ν(s,a))
    # Adjoint: dV_out[c'] = Σ_{s, a} dV_in[s] · P(a|s) · I(c' = ν(s,a))
    layout = s.input_layout
    cdim   = s.choice_dim
    actions = axisvalues(layout.axes[cdim])
    n_a     = length(actions)
    prob    = kernel.choice_prob
    dV_out  = zeros(T, size(dV_in))

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        d       = dV_in[ci_in]
        iszero(d) && continue
        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = s.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, cdim)
            dV_out[CartesianIndex(out_idxs)] += d * p
        end
    end
    return dV_out
end

function forward_adjoint!(s::LogitChoice{F,BF,LIn,LOut,N,D,T},
                          dΛ_end::AbstractArray{T,N},
                          kernel, scratch) where {F,BF,LIn,LOut,N,D,T}
    # Primal:  Λ_end[c'] = Σ_{s, a} P(a|s) · Λ_start[s] · I(c' = ν(s,a))
    # Adjoint: dΛ_start[s] = Σ_a P(a|s) · dΛ_end[ν(s,a)]
    layout = s.input_layout
    cdim   = s.choice_dim
    actions = axisvalues(layout.axes[cdim])
    n_a     = length(actions)
    prob    = kernel.choice_prob
    dΛ_start = zeros(T, size(dΛ_end))

    for (idx, cell) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        acc = zero(T)
        for (a_i, action) in pairs(actions)
            p = prob[in_idxs..., a_i]
            iszero(p) && continue
            next_axis_i = s.next_state_idx(cell, action)
            out_idxs    = Base.setindex(in_idxs, next_axis_i, cdim)
            acc += p * dΛ_end[CartesianIndex(out_idxs)]
        end
        dΛ_start[ci_in] = acc
    end
    return dΛ_start
end

# Migration #
#-----------#
# Same structure as LogitChoice, but the destination is the action index
# directly (no `next_state_idx` indirection).

"""CLAUDE
Reverse-mode backward-adjoint for [`Migration`](@ref): mirror of the
LogitChoice adjoint, with destination = action-index identity.
"""
function backward_adjoint!(stage::Migration{Cmat,T,N},
                           dV_in::AbstractArray{T,N},
                           kernel, scratch) where {Cmat,T,N}
    layout = stage.input_layout
    ldim   = stage.location_dim
    n_loc  = axissize(layout.axes[ldim])
    prob   = kernel.choice_prob
    dV_out = zeros(T, size(dV_in))
    dims   = layout_size(layout)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        d = dV_in[ci]
        iszero(d) && continue
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, ldim)
            dV_out[CartesianIndex(out_idxs)] += d * p
        end
    end
    return dV_out
end

"""CLAUDE
Reverse-mode forward-adjoint for [`Migration`](@ref): mirror of the
LogitChoice adjoint.
"""
function forward_adjoint!(stage::Migration{Cmat,T,N},
                          dΛ_end::AbstractArray{T,N},
                          kernel, scratch) where {Cmat,T,N}
    layout = stage.input_layout
    ldim   = stage.location_dim
    n_loc  = axissize(layout.axes[ldim])
    prob   = kernel.choice_prob
    dΛ_start = zeros(T, size(dΛ_end))
    dims     = layout_size(layout)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        acc = zero(T)
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, ldim)
            acc += p * dΛ_end[CartesianIndex(out_idxs)]
        end
        dΛ_start[ci] = acc
    end
    return dΛ_start
end

# ConsumptionSavings #
#--------------------#
# Structure identical to Argmax — a sparse-permutation kernel where the
# "policy" is the next-wealth grid index per input cell. Implemented
# for SSJ's `expectation_vectors` on the 3-stage chain
# `IncomeShock ∘ₛ IncomeReceipt ∘ₛ ConsumptionSavings`.

"""CLAUDE
Reverse-mode forward-adjoint for `ConsumptionSavings`: K is the
sparse permutation defined by the policy, so `K^T · dΛ_end` is a
gather along the wealth axis at each cell's stored policy index.
"""
function forward_adjoint!(stage::ConsumptionSavings{F_u,LIn,LOut,T,N},
                          dΛ_end::AbstractArray{T,N},
                          kernel, scratch) where {F_u,LIn,LOut,T,N}
    layout = stage.input_layout
    wdim   = stage.wealth_dim
    policy = stage.policy
    dΛ_start = similar(dΛ_end)

    for (idx, _) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        a1_i    = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a1_i, wdim)
        dΛ_start[ci_in] = dΛ_end[CartesianIndex(out_idxs)]
    end
    return dΛ_start
end

"""CLAUDE
Reverse-mode backward-adjoint for `ConsumptionSavings`: dual of the
above gather is a scatter `dV_out[c'] += dV_in[s]` along the policy.
"""
function backward_adjoint!(stage::ConsumptionSavings{F_u,LIn,LOut,T,N},
                           dV_in::AbstractArray{T,N},
                           kernel, scratch) where {F_u,LIn,LOut,T,N}
    layout = stage.input_layout
    wdim   = stage.wealth_dim
    policy = stage.policy
    dV_out = zeros(T, size(dV_in))

    for (idx, _) in cells(layout)
        in_idxs = Tuple(idx)
        ci_in   = CartesianIndex(in_idxs)
        a1_i    = policy[ci_in]
        out_idxs = Base.setindex(in_idxs, a1_i, wdim)
        dV_out[CartesianIndex(out_idxs)] += dV_in[ci_in]
    end
    return dV_out
end

# WealthChange #
#--------------#
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
K-transpose gather for [`WealthChange`](@ref): apply linear
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
Reverse-mode forward-adjoint for [`WealthChange`](@ref): K-transpose
applied to `dΛ_end` is a per-source-cell gather from `wgrid` via
share-based linear interpolation at the materialised `wpost` value.

Iterates over slices along the wealth axis (the kernel stores `wpost`
as a per-cell N-D array; the other axes are pass-through).
"""
function forward_adjoint!(stage::WealthChange{F,T,N,D,L,AV,Extrap},
                          dΛ_end::AbstractArray{T,N},
                          kernel, scratch) where {F,T,N,D,L,AV,Extrap}
    layout = stage.input_layout
    wdim   = stage.wealth_dim
    wgrid  = collect(T, axisvalues(layout.axes[wdim]))
    wpost  = kernel.wealth_post

    dΛ_start = similar(dΛ_end)
    dims = size(dΛ_end)
    other = ntuple(i -> i == wdim ? 1 : dims[i], N)

    if wdim == 1
        for other_ci in CartesianIndices(other)
            tail = other_ci.I[2:end]
            dΛ_start_view = view(dΛ_start, :, tail...)
            dΛ_end_view   = view(dΛ_end,   :, tail...)
            wpost_view    = view(wpost,    :, tail...)
            _share_gather!(dΛ_start_view, dΛ_end_view, wpost_view, wgrid)
        end
    else
        # General permutation: same approach as in `_along_wealth`.
        perm     = _bring_dim_first(N, wdim)
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
Reverse-mode backward-adjoint for [`WealthChange`](@ref): not yet
implemented. The primal `backward!` interpolates `V_end` linearly at
`wpost(cell)`, so the dual would be a share-based scatter — exactly
the structure of `convert_distribution!`. SSJ's `expectation_vectors`
pipeline does not need this method (it only uses `forward_adjoint!`),
so it is deliberately stubbed for now. If reverse-mode gradients
through the value function are needed later, mirror the `_share_gather!`
helper with a scatter pattern.
"""
function backward_adjoint!(stage::WealthChange{F,T,N,D,L,AV,Extrap},
                           dV_start::AbstractArray{T,N},
                           kernel, scratch) where {F,T,N,D,L,AV,Extrap}
    error("WealthChange.backward_adjoint! not yet implemented. " *
          "Only `forward_adjoint!` (used by `expectation_vectors`) is " *
          "currently wired up; the backward adjoint would mirror the " *
          "share-based gather as a scatter — add when reverse-mode " *
          "gradients through V are needed.")
end

