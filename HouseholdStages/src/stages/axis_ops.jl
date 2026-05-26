# Axis-management helpers shared across stages. Small, allocation-free
# primitives that show up in more than one stage body (permutation
# tuples, singleton-axis insertion, name-keyed slicing). Library-internal;
# nothing here is exported.

"""
Permutation tuple bringing axis `dim` to position 1 while keeping
the relative order of the other axes.
"""
_bring_dim_first(N::Int, dim::Int) =
    ntuple(i -> i == 1 ? dim : (i <= dim ? i - 1 : i), N)

function _insert_singleton(dims::NTuple{N, Int}, pos::Int) where {N}
    @assert 1 <= pos <= N + 1
    return ntuple(i -> i < pos  ? dims[i]   :
                       i == pos ? 1         :
                                  dims[i-1], Val(N + 1))
end

# Named-axis helpers #
#--------------------#
# Thin name-keyed wrappers over `eachslice`, `selectdim`, `permutedims`,
# and `reshape`. The intent is to keep `StateLayout`'s named-axis
# abstraction visible inside stage bodies instead of erasing it to
# integer dims immediately. None of these allocate (the permute helpers
# take a preallocated `dest`).

"""
Integer dimension number of axis `axis` in `layout`. Pass-through to
`axis_position`; kept for vocabulary uniformity at call sites that
read more naturally as "the array's axis dim" than "the layout's axis
position".
"""
axis_dim(layout::StateLayout, axis::Symbol) = axis_position(layout, axis)

"""
Iterator of 1-D slices of `A` along the named `axis`. Wraps
`eachslice(A; dims=axis_dim(layout, axis))`. Each yielded slice is a
view into `A` over the named axis with all other axes fixed at one
cartesian index.
"""
slices_along(A::AbstractArray, layout::StateLayout, axis::Symbol) =
    eachslice(A; dims=axis_dim(layout, axis))

"""
Iterator of slices of `A` that fix the named `axis` as free and
collapse every other axis to a scalar index. Equivalent to
`eachslice(A; dims=setdiff(1:ndims(A), axis_dim(layout, axis)))`: each
yielded slice is the 1-D view obtained by holding the named axis and
running over a cartesian sweep of all the other axes.

`@inline` + `Val`-keyed `ntuple` length so the returned `Slices` type
is fully concrete when called from a context where the layout's
`Names` are statically known (the usual stage-body case). Without
this, the resulting `Slices` carries an `NTuple{N-1, Int}` whose
elements are runtime values and each per-slice view escapes to the
heap — observed as ~760 allocations per backward in `wealth_change`
before this annotation.
"""
@inline function slices_over(A::AbstractArray, layout::StateLayout, axis::Symbol)
    d = axis_dim(layout, axis)
    return eachslice(A; dims=ntuple(i -> i < d ? i : i + 1, Val(ndims(A) - 1)))
end

"""
Name-keyed `selectdim`: return the view of `A` obtained by fixing the
named `axis` at integer index `i`. The pair syntax (`:axis => i`)
mirrors the `Dict`-style pattern used elsewhere.
"""
fix(A::AbstractArray, layout::StateLayout, (axis, i)::Pair{Symbol, <:Integer}) =
    selectdim(A, axis_dim(layout, axis), i)

"""
Return `A` with the named `axis` brought to position 1. When the axis
is already first this is a no-op and returns `A` unchanged (same
object via `===`); otherwise allocates a fresh permuted copy via
`permutedims`. Callers must not assume the return is always a fresh
array.
"""
function permute_to_first(A::AbstractArray, layout::StateLayout, axis::Symbol)
    d = axis_dim(layout, axis)
    d == 1 && return A
    return permutedims(A, _bring_dim_first(ndims(A), d))
end

"""
In-place version of `permute_to_first`: writes the permuted array into
the preallocated `dest`. When the named `axis` is already first this
is a no-op and returns `A` unchanged (same object via `===`) without
touching `dest`; otherwise calls `permutedims!(dest, A, perm)` and
returns `dest`.
"""
function permute_to_first!(dest::AbstractArray, A::AbstractArray, layout::StateLayout, axis::Symbol)
    d = axis_dim(layout, axis)
    d == 1 && return A
    return permutedims!(dest, A, _bring_dim_first(ndims(A), d))
end

"""
Reshape `A` inserting a singleton at the named `axis`'s position.
Generalises `_insert_singleton` to the name-keyed surface: lifts an
`(N)`-D array into an `(N+1)`-D array with size 1 along `axis`. Useful
for broadcasting an aggregate (the `forgetful_sum` style) back across
a dropped axis.
"""
function with_singleton(A::AbstractArray, layout::StateLayout, axis::Symbol)
    pos = axis_dim(layout, axis)
    return reshape(A, _insert_singleton(size(A), pos))
end

"""
Return a new `CartesianIndex` equal to `ci` but with the coordinate at
the named `axis`'s position replaced by `j`. The pair syntax mirrors
`fix`; useful when building `next_ci` caches (`argmax`,
`logit_choice`) at construction time.
"""
set_coord(ci::CartesianIndex, layout::StateLayout, (axis, j)::Pair{Symbol, <:Integer}) =
    CartesianIndex(Base.setindex(Tuple(ci), j, axis_dim(layout, axis)))
