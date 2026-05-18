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
    continuous_grid(lo, hi; size) -> ContinuousGrid
    continuous_grid(grid::AbstractVector) -> ContinuousGrid

Construct a continuous-grid axis kind. The two-argument form builds an
evenly-spaced grid of `size` points on `[lo, hi]`; the one-argument form
wraps a user-supplied numeric vector directly.
"""
continuous_grid(lo::Real, hi::Real; size::Int) =
    ContinuousGrid(collect(range(lo, hi; length = size)))
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
    StateAxis{Name, K<:AxisKind}

A named axis of a state layout. `Name` is a `Symbol` carried in the type
parameter so axis names participate in compile-time dispatch and
NamedTuple-keyed iteration (`cells`). `kind` carries the axis's runtime
data (grid or levels).

Construct via `StateAxis(name, kind)`.
"""
struct StateAxis{Name, K<:AxisKind}
    kind :: K
end

StateAxis(name::Symbol, kind::K) where {K<:AxisKind} =
    StateAxis{name, K}(kind)

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
    remaining = ntuple(i -> i < pos ? layout.axes[i] : layout.axes[i+1],
                       length(layout.axes) - 1)
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
axis_position(::Type{L}, name::Symbol) where {L<:StateLayout} =
    axis_position(L, Val(name))
function axis_position(::Type{<:StateLayout{Names}}, ::Val{name}) where {Names, name}
    idx = findfirst(==(name), Names)
    idx === nothing && error("axis :$name not found in layout with names $(Names)")
    return idx
end

# Cells iteration #
#-----------------#

"""
    CellsIterator{Names, V<:Tuple, N}

Iterator returned by [`cells`](@ref). Each iteration yields a pair
`(idx, cell)`:

  * `idx`  — `NamedTuple{Names}` of `Int` axis indices.
  * `cell` — `NamedTuple{Names}` of axis-coordinate values (leaves typed
    `Float64`, `Int`, `Symbol`, …).
"""
struct CellsIterator{Names, V<:Tuple, N}
    values :: V
    sizes  :: NTuple{N, Int}
end

"""
    cells(layout) -> CellsIterator

Iterate the cells of a [`StateLayout`](@ref) in column-major order. Each
iteration yields `(idx, cell)`: an integer-index NamedTuple and an
axis-value NamedTuple, both keyed by axis name.
"""
function cells(layout::StateLayout{Names}) where {Names}
    N = length(layout.axes)
    values = map(axisvalues, layout.axes)
    sizes  = map(axissize,   layout.axes)
    return CellsIterator{Names, typeof(values), N}(values, sizes)
end

Base.length(it::CellsIterator) = prod(it.sizes)
Base.IteratorSize(::Type{<:CellsIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CellsIterator}) = Base.HasEltype()
Base.eltype(::Type{<:CellsIterator{Names}}) where {Names} = Tuple

function Base.iterate(it::CellsIterator{Names, V, N}) where {Names, V, N}
    ci = CartesianIndices(it.sizes)
    nxt = iterate(ci)
    nxt === nothing && return nothing
    cart, ci_state = nxt
    return _cells_pair(it, cart), (ci, ci_state)
end

function Base.iterate(it::CellsIterator{Names, V, N}, state) where {Names, V, N}
    ci, ci_state = state
    nxt = iterate(ci, ci_state)
    nxt === nothing && return nothing
    cart, ci_state2 = nxt
    return _cells_pair(it, cart), (ci, ci_state2)
end

function _cells_pair(it::CellsIterator{Names, V, N},
                     cart::CartesianIndex{N}) where {Names, V, N}
    idxs = cart.I
    idx_nt  = NamedTuple{Names}(idxs)
    cell_nt = NamedTuple{Names}(ntuple(i -> it.values[i][idxs[i]], Val(N)))
    return (idx_nt, cell_nt)
end

# Broadcastable cell array #
#--------------------------#

"""CLAUDE
Return an `N`-D dense `Array{NamedTuple}` of cell values, shape =
`layout_size(layout)`. Each entry is the axis-value `NamedTuple` for that
multi-index. Suitable as a broadcast operand for closures of the form
`f(cell, args...; env)`: `f.(cell_array(layout), other_arr; env=Ref(env))`
evaluates `f` at each cell elementwise.

The element type is `NamedTuple{Names, Tuple{...}}` and is isbits when
every axis stores isbits values (typically the case for numeric axes;
`Symbol`-valued categorical axes break isbits and require CPU broadcasts).
"""
function cell_array(layout::StateLayout{Names}) where {Names}
    N = length(layout.axes)
    values = map(axisvalues, layout.axes)
    sizes  = map(axissize,   layout.axes)
    cells_nt = Array{_cell_eltype(layout)}(undef, sizes)
    for cart in CartesianIndices(sizes)
        idxs = cart.I
        cells_nt[cart] = NamedTuple{Names}(ntuple(i -> values[i][idxs[i]], Val(N)))
    end
    return cells_nt
end

# Type of one cell in `cell_array(layout)`. Used for storage typing.
function _cell_eltype(layout::StateLayout{Names}) where {Names}
    value_eltypes = map(a -> eltype(axisvalues(a)), layout.axes)
    return NamedTuple{Names, Tuple{value_eltypes...}}
end
