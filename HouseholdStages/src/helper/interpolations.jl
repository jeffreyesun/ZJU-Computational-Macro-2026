##############################
# Interpolation Helpers      #
##############################

# Routines used by the value-function (V) backward pass and the
# distribution (Λ) forward pass when a stage performs a deterministic
# wealth change or a continuous savings choice. Adapted from
# `reference_materials/example_stages/helper/interpolations.jl` with two
# user-requested modifications to `convert_distribution!`:
#
#   1. The hard `iszero(y[end])` precondition on the source distribution
#      is dropped — `y[end]` may be any non-negative value (e.g., the
#      overflow bin of a prior call).
#   2. When source mass would land past the destination grid's right end
#      (`x[i] >= x_new[end]`), it accumulates into `y_new[end]` (the last
#      destination bin), not the second-to-last bin.
#
# Underflow (source mass at `x[i] < x_new[1]`) symmetrically accumulates
# into `y_new[1]`.

# Linear interpolation of V #
#---------------------------#

"""CLAUDE
Linearly interpolate `(x1, y1)` onto `x2`, writing into `y2`. Both grids
must be sorted. `extrap` controls left-extrapolation when `x2[i] < x1[1]`:
`:linear` extends the leftmost slope, `:clip` snaps to `y1[1]`, and
`-Inf` marks the point unreachable (used to flag infeasible continuation
states by [`WealthChange`](@ref) backward passes).
"""
function reinterpolate!(y2::AbstractVector, y1::AbstractVector,
                        x1::AbstractVector, x2::AbstractVector,
                        ::Val{extrap}) where extrap
    j = 1
    x1_j1 = x1[2]
    len_x2 = length(x2)
    len_x1m1 = length(x1) - 1

    i0 = 1
    # Left-extrapolation branch: clip or -Inf for x2[i] < x1[1].
    if extrap != :linear
        while x2[i0] < x1[1]
            y2[i0] = extrap == :clip ? y1[1] : -Inf
            i0 == len_x2 && return y2
            i0 += 1
        end
    end

    for i in i0:len_x2
        x2_i = x2[i]
        while x2_i > x1_j1
            j == len_x1m1 && break
            j += 1
            x1_j1 = x1[j+1]
        end
        x1_j = x1[j]
        y1_j = y1[j]
        y1_j1 = y1[j+1]
        if y1_j == -Inf || y1_j1 == -Inf
            y2[i] = extrap == :clip ? max(y1_j, y1_j1) : -Inf
        else
            slope = (y1_j1 - y1_j) / (x1_j1 - x1_j)
            y2[i] = slope * (x2_i - x1_j) + y1_j
        end
    end
    return y2
end

"""CLAUDE
Apply [`reinterpolate!`](@ref) along the leading dimension of
arbitrary-rank arrays, broadcasting `x1` / `x2` across trailing dims
when they have length 1 there.
"""
function reinterpolate_arr!(y2, y1, x1, x2, ::Val{extrap}) where extrap
    for idx in CartesianIndices(Base.tail(size(y2)))
        _tview(arr) = @view arr[:, _broadcast_index(idx, Base.tail(size(arr)))]
        reinterpolate!(_tview(y2), _tview(y1), _tview(x1), _tview(x2), Val(extrap))
    end
    return y2
end

# Distribution conversion #
#-------------------------#

"""CLAUDE
Convert a non-negative weight vector `y` on grid `x` to a weight vector
`y_new` on grid `x_new`, preserving total mass.

`interp == :share` splits each source mass between adjacent destination
gridpoints by linear weight; `:rounddown` places each source mass
entirely on the nearest destination gridpoint at or below `x[i]`. Both
grids must be sorted. Underflow (`x[i] < x_new[1]`) accumulates into
`y_new[1]`; overflow (`x[i] >= x_new[end]`) accumulates into `y_new[end]`.
"""
function convert_distribution!(y_new::AbstractVector, y::AbstractVector,
                               x::AbstractVector, x_new::AbstractVector,
                               ::Val{interp} = Val(:share)) where interp
    fill!(y_new, zero(eltype(y_new)))
    len_x = length(x)
    len_x_new_minus_1 = length(x_new) - 1

    # Monotone walk over destination bins.
    j = 1
    for i in 1:len_x
        x_i = x[i]
        y_i = y[i]
        iszero(y_i) && continue

        # Underflow.
        if x_i < x_new[1]
            y_new[1] += y_i
            continue
        end

        # Overflow — all subsequent x_k are also at or past x_new[end].
        if x_i >= x_new[end]
            y_new[end] += y_i
            for k in (i+1):len_x
                y_new[end] += y[k]
            end
            return y_new
        end

        # Advance the destination cursor until x_new[j] <= x_i < x_new[j+1].
        while x_i >= x_new[j+1]
            j += 1
            j == len_x_new_minus_1 && break
        end

        if interp == :share
            left_share = (x_new[j+1] - x_i) / (x_new[j+1] - x_new[j])
            y_new[j]   += y_i * left_share
            y_new[j+1] += y_i * (1 - left_share)
        elseif interp == :rounddown
            y_new[j] += y_i
        else
            error("convert_distribution!: invalid interp option $interp")
        end
    end
    return y_new
end

"""CLAUDE
Apply [`convert_distribution!`](@ref) along the leading dimension of
arbitrary-rank arrays, broadcasting `x` / `x_new` across trailing dims
when they have length 1 there.
"""
function convert_distribution_arr!(y_new, y, x, x_new,
                                   ::Val{interp} = Val(:share)) where interp
    for idx in CartesianIndices(Base.tail(size(y_new)))
        _tview(arr) = @view arr[:, _broadcast_index(idx, Base.tail(size(arr)))]
        convert_distribution!(_tview(y_new), _tview(y),
                              _tview(x), _tview(x_new), Val(interp))
    end
    return y_new
end

# Monotone-policy argmax #
#------------------------#

"""CLAUDE
Maximize `u_slice[a, s] + V_post[a]` over `a` for each `s`, with the
monotone-policy guarantee that the optimum is non-decreasing in `s`. The
search for each `s` therefore starts where `s-1`'s search ended, so the
total work is `O(N + M)` rather than `O(N M)`.

Writes the integer argmax into `policy_slice[s]` and the resulting
maximum into `V_prec_slice[s]`. Returns `policy_slice`.

If no action is feasible (all candidates give `-Inf`), `policy_slice[s]
= 1`, `V_prec_slice[s] = -Inf`, and the monotone lower bound is *not*
advanced — that cell does not contaminate downstream cells' search.
"""
function k1_argmax_monotone!(V_prec_slice::AbstractVector{T},
                             policy_slice::AbstractVector{Int},
                             u_slice::AbstractMatrix{T},
                             V_post::AbstractVector{T}) where T
    n_s = length(V_prec_slice)
    n_a = length(V_post)
    @assert size(u_slice) == (n_a, n_s)
    @assert length(policy_slice) == n_s

    prev_a = 1
    for s in 1:n_s
        best_v = typemin(T)
        best_a = 0
        for a in prev_a:n_a
            u = u_slice[a, s]
            isfinite(u) || continue
            v = u + V_post[a]
            if v > best_v
                best_v = v
                best_a = a
            end
        end
        if best_a == 0
            V_prec_slice[s] = typemin(T)
            policy_slice[s] = 1
        else
            V_prec_slice[s] = best_v
            policy_slice[s] = best_a
            prev_a = best_a
        end
    end
    return policy_slice
end

# Monotone-policy divide-and-conquer argmax #
#-------------------------------------------#

"""CLAUDE
Divide-and-conquer monotone-policy argmax. Same problem as
[`k1_argmax_monotone!`](@ref): for each `s`, maximise `u_slice[a, s] +
V_post[a]` over `a`, with the monotone-policy guarantee that the optimum
is non-decreasing in `s`. The D&C variant exploits the monotonicity by
tightening *both* the lower and upper bounds at each midpoint, achieving
`O((n_a + n_s) log n_s)` total work — strictly better than the sequential
walk's `O(n_a + n_s · max_window_size)` worst case.

Adapted from `reference_materials/example_stages/helper/interpolations.jl`'s
`k1_argmax!`, generalized to support arbitrary `n_s` (the reference's
iterative variant required `ispow2(n_s - 1)` and left non-power-of-two
sizes for the recursive form, which is what we use here).

**Caveat.** Correct iff the optimum policy is non-decreasing in `s` —
in the consumption-savings setting this is the always-non-negative-MPS
assumption (concave utility plus linear budget implies it, but it can
fail under non-convex feasibility, additive lotteries, or hand-built
non-concave payoffs). Prefer the sequential walk when in doubt.
"""
function k1_argmax_dc!(V_prec_slice::AbstractVector{T},
                       policy_slice::AbstractVector{Int},
                       u_slice::AbstractMatrix{T},
                       V_post::AbstractVector{T}) where T
    n_s = length(V_prec_slice)
    n_a = length(V_post)
    @assert size(u_slice) == (n_a, n_s)
    @assert length(policy_slice) == n_s
    n_s == 0 && return policy_slice
    _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                        1, n_s, 1, n_a)
    return policy_slice
end

# Fill `V_prec_slice[lo..hi]` / `policy_slice[lo..hi]` given that each
# entry's optimum is in `[lo_b..hi_b]`. The midpoint's argmax bounds
# the search range of its left and right sub-problems.
function _dc_argmax_recurse!(V_prec_slice::AbstractVector{T},
                             policy_slice::AbstractVector{Int},
                             u_slice::AbstractMatrix{T},
                             V_post::AbstractVector{T},
                             lo::Int, hi::Int,
                             lo_b::Int, hi_b::Int) where T
    lo > hi && return
    mid = (lo + hi) >> 1
    best_v = typemin(T)
    best_a = 0
    for a in lo_b:hi_b
        u = u_slice[a, mid]
        isfinite(u) || continue
        v = u + V_post[a]
        if v > best_v
            best_v = v
            best_a = a
        end
    end
    if best_a == 0
        # No feasible action at this midpoint; downstream cells still
        # respect the outer bounds (we don't shrink to a missing pivot).
        V_prec_slice[mid] = typemin(T)
        policy_slice[mid] = 1
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            lo, mid - 1, lo_b, hi_b)
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            mid + 1, hi, lo_b, hi_b)
    else
        V_prec_slice[mid] = best_v
        policy_slice[mid] = best_a
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            lo, mid - 1, lo_b, best_a)
        _dc_argmax_recurse!(V_prec_slice, policy_slice, u_slice, V_post,
                            mid + 1, hi, best_a, hi_b)
    end
    return
end

# Shared utility — broadcast index mapping #
#------------------------------------------#

"Collapse any dim of length 1 (or beyond `len(size)`) in `idx` to 1, mimicking broadcasting semantics."
function _broadcast_index(idx::CartesianIndex, size::Tuple)
    return CartesianIndex(ntuple(
        i -> i > length(size) ? 1 : min(idx[i], size[i]),
        Val(length(idx))))
end
