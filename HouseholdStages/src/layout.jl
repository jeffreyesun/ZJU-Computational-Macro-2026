"""
    AxisKind

Abstract supertype for state-axis kinds. Two concrete kinds are supported:

  * [`ContinuousGrid`](@ref) — a numeric grid used for interpolation
    (continuous-state axes: wealth, productivity, …).
  * [`DiscreteFinite`](@ref) — a finite vector of levels. Leaf type may be
    `Float64`, `Int`, `Symbol`, or any other `Bits`-typed value.

Construct via the helper functions [`continuous_grid`](@ref),
[`discrete_finite`](@ref), and [`categorical`](@ref).
"""
abstract type AxisKind end

"""
    ContinuousGrid{T<:Real, V<:AbstractVector{T}} <: AxisKind

A continuous axis represented by a numeric grid `grid::V`. Used for
interpolation along the axis.
"""
struct ContinuousGrid{T<:Real, V<:AbstractVector{T}} <: AxisKind
    grid :: V
end

"""
    DiscreteFinite{T, V<:AbstractVector{T}} <: AxisKind

A discrete axis represented by a finite vector of levels `levels::V`. Leaf
type `T` may be `Float64`, `Int`, `Symbol`, or any other `Bits`-typed value.
"""
struct DiscreteFinite{T, V<:AbstractVector{T}} <: AxisKind
    levels :: V
end

"""
    continuous_grid(lo, hi; length, spacing = :linear) -> ContinuousGrid
    continuous_grid(grid::AbstractVector)             -> ContinuousGrid

Construct a continuous-grid axis kind. The two-argument form builds a
grid of `length` points on `[lo, hi]` with the requested spacing — either
`:linear` (`range(lo, hi; length)`) or `:log` (geometric in `(lo-shift,
hi-shift)`, dense near `lo` and coarse near `hi`, matching the wealth-
grid convention used in `examples/`). The one-argument form wraps a
user-supplied numeric vector directly. The kwarg `size` is also
accepted as a deprecated synonym for `length`.

`:log` spacing maps `t ∈ [0, log((hi-lo+shift)/shift)]` through
`exp(t)·shift - shift + lo`, putting the first knot at `lo`, the last
at `hi`, and concentrating points near `lo`. `shift` defaults to `1`.
"""
function continuous_grid(lo::Real, hi::Real;
                         length::Union{Nothing, Int} = nothing,
                         size::Union{Nothing, Int}   = nothing,
                         spacing::Symbol             = :linear,
                         shift::Real                 = 1.0)
    n = @something length size error("continuous_grid: pass either `length` or `size`")
    if spacing === :linear
        return ContinuousGrid(collect(range(lo, hi; length = n)))
    elseif spacing === :log
        grid = [exp(t) * shift - shift + lo
                for t in range(0.0, log((hi - lo + shift) / shift); length = n)]
        return ContinuousGrid(grid)
    else
        error("continuous_grid: spacing must be :linear or :log, got :$spacing")
    end
end
continuous_grid(grid::AbstractVector{<:Real}) = ContinuousGrid(grid)

"""
    discrete_finite(levels::AbstractVector) -> DiscreteFinite

Construct a discrete-finite axis kind from a vector of levels. The leaf
type is inferred from the vector's element type.
"""
discrete_finite(levels::AbstractVector{T}) where {T} =
    DiscreteFinite{T, typeof(levels)}(levels)

"""
    categorical(syms::AbstractVector{Symbol}) -> DiscreteFinite{Symbol}

Sugar over [`discrete_finite`](@ref) for symbolic categorical axes.
"""
categorical(syms::AbstractVector{Symbol}) = discrete_finite(syms)

"""
    StateAxis(name, kind) -> StateAxis{name, typeof(kind)}

A named axis of a state layout. `Name` is a `Symbol` carried in the type
parameter so axis names participate in compile-time dispatch and
NamedTuple-keyed iteration (`cells`). `kind` carries the axis's runtime
data (grid or levels).
"""
struct StateAxis{Name, K<:AxisKind}
    kind :: K
end

StateAxis(name::Symbol, kind::K) where {K<:AxisKind} =
    StateAxis{name, K}(kind)

"""
    StateAxis(name, levels::AbstractVector) -> StateAxis

Shortcut: a raw `AbstractVector` is treated as a [`DiscreteFinite`](@ref)
level set. Equivalent to `StateAxis(name, discrete_finite(levels))`.
Use [`continuous_grid`](@ref) explicitly for continuous (interpolated)
axes.
"""
StateAxis(name::Symbol, levels::AbstractVector) =
    StateAxis(name, discrete_finite(levels))

"""
    axisname(axis) -> Symbol

Return the axis's name (the `Name` type parameter of [`StateAxis`](@ref)).
"""
axisname(::StateAxis{Name}) where {Name} = Name
axisname(::Type{<:StateAxis{Name}}) where {Name} = Name

"""
    axissize(axis_or_kind) -> Int

Number of cells along the axis.
"""
axissize(a::StateAxis) = axissize(a.kind)
axissize(k::ContinuousGrid) = length(k.grid)
axissize(k::DiscreteFinite) = length(k.levels)

"""
    axisvalues(axis_or_kind) -> AbstractVector

The vector of axis-coordinate values (grid points or levels).
"""
axisvalues(a::StateAxis) = axisvalues(a.kind)
axisvalues(k::ContinuousGrid) = k.grid
axisvalues(k::DiscreteFinite) = k.levels

"""
    StateLayout{Names, Axes<:Tuple}

An ordered, named layout of state axes. `Names` is the tuple of axis
names (lifted to the type level so that NamedTuple-keyed iteration is
type-stable). `Axes` is the concrete tuple type of [`StateAxis`](@ref)
instances.

Construct via `StateLayout(axes...)`.
"""
struct StateLayout{Names, Axes<:Tuple}
    axes :: Axes
end

function StateLayout(axes::StateAxis...)
    names = map(axisname, axes)
    if length(unique(names)) != length(names)
        error("StateLayout axis names must be unique; got $(names)")
    end
    return StateLayout{names, typeof(axes)}(axes)
end

"""
    drop_axis(layout, name::Symbol) -> StateLayout

Return a new layout with the axis named `name` removed.
"""
function drop_axis(layout::StateLayout, name::Symbol)
    pos = axis_position(layout, name)
    remaining = (layout.axes[1:pos-1]..., layout.axes[pos+1:end]...)
    return StateLayout(remaining...)
end

"""
    axisnames(layout) -> NTuple{N, Symbol}

The names of `layout`'s axes, in order.
"""
axisnames(::StateLayout{Names}) where {Names} = Names
axisnames(::Type{<:StateLayout{Names}}) where {Names} = Names

Base.length(layout::StateLayout) = length(layout.axes)

"""
    layout_size(layout) -> NTuple{N, Int}

Per-axis sizes, in order.
"""
layout_size(layout::StateLayout) = map(axissize, layout.axes)

"""
    axis_position(layout, name::Symbol) -> Int

Integer dimension number of the axis named `name` in `layout`. Errors if
no axis has that name.
"""
function axis_position(layout::StateLayout, name::Symbol)
    names = axisnames(layout)
    idx = findfirst(==(name), names)
    idx === nothing && error("axis :$name not found in layout with names $(names)")
    return idx
end

# Cells iteration #
#-----------------#

# Type-stable single-cell builder: out-of-line so the closure capturing
# `values` (a heterogeneous tuple) doesn't poison inference inside the
# generator returned by `cells`.
function _cell_pair(values::V, ci::CartesianIndex{N}, ::Val{Names}) where {Names, V, N}
    idx_nt  = NamedTuple{Names}(Tuple(ci))
    cell_nt = NamedTuple{Names}(ntuple(i -> values[i][ci[i]], Val(N)))
    return (idx_nt, cell_nt)
end

"""
    cells(layout) -> generator

Iterate the cells of a [`StateLayout`](@ref) in column-major order. Each
iteration yields `(idx, cell)`: an integer-index NamedTuple and an
axis-value NamedTuple, both keyed by axis name.
"""
function cells(layout::StateLayout{Names}) where {Names}
    values = map(axisvalues, layout.axes)
    sizes  = layout_size(layout)
    return (_cell_pair(values, ci, Val(Names)) for ci in CartesianIndices(sizes))
end

# Broadcastable cell array #
#--------------------------#

"""CLAUDE
Return an `N`-D dense `Array{NamedTuple}` of cell values, shape =
`layout_size(layout)`. Each entry is the axis-value `NamedTuple` for that
multi-index. Suitable as a broadcast operand for closures of the form
`f(cell; env)`: `f.(cell_array(layout); env)` evaluates `f` at each cell
elementwise (`env` is passed plainly — broadcasting treats a non-array
scalar as constant).

The element type is `NamedTuple{Names, Tuple{...}}` and is isbits when
every axis stores isbits values (typically the case for numeric axes;
`Symbol`-valued categorical axes break isbits and require CPU broadcasts).
"""
function cell_array(layout::StateLayout{Names}) where {Names}
    N = length(layout.axes)
    values = map(axisvalues, layout.axes)
    sizes  = layout_size(layout)
    return [NamedTuple{Names}(ntuple(i -> values[i][cart[i]], Val(N)))
            for cart in CartesianIndices(sizes)]
end
