"""
A deterministic wealth-change stage. Each cell's wealth transitions
according to the user closure `wealth_post(cell; env) -> Real`. The
K-operator is "look up the value at `wealth_post(cell; env)` along the
wealth axis": backward uses linear interpolation on `V_end`, forward
uses share-based mass redistribution. Kernel = the materialised
`wealth_post` value per cell.

Other state axes pass through unchanged. The `:wealth` axis is
expected to be a [`ContinuousGrid`](@ref); other axes can be of any
kind.

Two extrapolation policies are wired up via the `extrap` kwarg for the
backward V-interpolation:

  * `:linear` — extend the leftmost slope into the underflow region.
  * `:clip`   — snap underflow to `V_end[1, ...]`.
  * `-Inf`    — mark underflow cells as unreachable (useful for
    enforcing borrowing constraints).

Underflow / overflow on the forward Λ-push are handled by
[`convert_distribution!`](@ref): mass landing below `wgrid[1]`
accumulates in `wgrid[1]`, mass landing past `wgrid[end]` accumulates in
`wgrid[end]`.
"""
struct WealthChange{F, T<:Real, N, D, L<:StateLayout,
                    AV<:AbstractArray{T,N}, Extrap} <: AbstractStage
    wealth_post   :: F
    wealth_axis   :: Symbol
    wealth_dim    :: Int
    closure_deps  :: NTuple{D, Symbol}
    input_layout  :: L
    output_layout :: L
    V_start       :: AV
    Λ_end         :: AV
end

"""
    WealthChange(layout; wealth_post, wealth_axis=:wealth, closure_deps=(),
                          extrap=:linear, element_type=Float64) -> WealthChange

Build a [`WealthChange`](@ref) stage. `wealth_post(cell; env)` returns
the post-stage wealth value as a function of the cell's pre-stage state
and the environment. `extrap` is `:linear` / `:clip` / `-Inf`.
"""
function WealthChange(layout::StateLayout;
                      wealth_post,
                      wealth_axis::Symbol = :wealth,
                      closure_deps::NTuple{D, Symbol} = (),
                      extrap = :linear,
                      element_type::Type{T} = Float64,
                      V_start::Union{Nothing, AbstractArray} = nothing,
                      Λ_end::Union{Nothing, AbstractArray}  = nothing) where {D, T<:Real}
    extrap in (:linear, :clip, -Inf) ||
        error("WealthChange: extrap must be :linear, :clip, or -Inf; got $extrap")
    wealth_dim = axis_position(layout, wealth_axis)
    dims = layout_size(layout)
    N    = length(dims)
    Vs   = @something V_start zeros(T, dims)
    Λe   = @something Λ_end   zeros(T, dims)
    @assert typeof(Vs) === typeof(Λe) "WealthChange: V_start and Λ_end must have the same concrete array type"
    return WealthChange{typeof(wealth_post), T, N, D, typeof(layout),
                        typeof(Vs), Val(extrap) |> typeof}(
        wealth_post, wealth_axis, wealth_dim, closure_deps,
        layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:WealthChange}) = NamedTuple()

# Kernel holds the materialised `wealth_post` array (same shape as the
# state layout). Scratch is the wgrid array reshaped along the wealth
# dim (so reinterpolate_arr! / convert_distribution_arr! can read it
# along their leading dim).
function allocate(stage::WealthChange{F,T,N,D,L,AV,Extrap},
                  ::Type{T2} = T) where {F,T,N,D,L,AV,Extrap,T2}
    dims        = layout_size(stage.input_layout)
    wealth_post = zeros(T2, dims)
    # Materialize the cell array once at workspace-allocation time. The
    # layout is a struct field and never changes, so the cells are
    # safe to cache for the lifetime of the workspace. This eliminates
    # the per-backward `cell_array(layout)` allocation in `_fill_wealth_post!`.
    cells = cell_array(stage.input_layout)
    return ((wealth_post = wealth_post,), (cells = cells,))
end

# Backward #
#----------#
# V_pre(cell) = V_post evaluated at wealth_post(cell; env), interpolated
# along the wealth axis using the chosen extrap policy.

function backward!(stage::WealthChange{F,T,N,D,L,AV,Extrap},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {F,T,N,D,L,AV,Extrap}
    layout = stage.input_layout
    wdim   = stage.wealth_dim
    wgrid  = axisvalues(layout.axes[wdim])
    wpost  = kernel.wealth_post

    _fill_wealth_post!(wpost, stage.wealth_post, scratch.cells, env)

    # V_end lives on wgrid; we want V_start at the per-cell query points wpost.
    extrap_val = _extrap_value(Extrap)
    _along_wealth(stage.V_start, V_end, wgrid, wpost, wdim,
                  (y2, y1, x1, x2) -> reinterpolate!(y2, y1, x1, x2, extrap_val))
    return stage.V_start
end

# Forward #
#---------#
# Λ_post on wgrid = convert_distribution(Λ_pre with source positions
# wpost → wgrid), via share-based redistribution.

function forward!(stage::WealthChange{F,T,N,D,L,AV,Extrap},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {F,T,N,D,L,AV,Extrap}
    layout = stage.input_layout
    wdim   = stage.wealth_dim
    wgrid  = axisvalues(layout.axes[wdim])
    wpost  = kernel.wealth_post
    # Λ_start has source positions wpost (per cell); Λ_end lives on wgrid.
    _along_wealth(stage.Λ_end, Λ_start, wpost, wgrid, wdim,
                  (y2, y1, x1, x2) -> convert_distribution!(y2, y1, x1, x2, Val(:share)))
    return stage.Λ_end
end

# Internals #
#-----------#

# Evaluate wealth_post at every cell; populate `wpost`. The closure is
# `wealth_post(cell; env)` — broadcast over the cached cell array
# (built once per `allocate` call and stored in scratch). Functions
# broadcast as scalars by default in Julia, so the broadcast iterates
# cells while passing `Ref(env)` as a scalar kwarg.
function _fill_wealth_post!(wpost, wealth_post,
                            cells_arr::AbstractArray, env)
    wpost .= wealth_post.(cells_arr; env = Ref(env))
    return wpost
end

# Walk the wealth axis at every (other-axis) slice and call `op(y2, y1,
# x1, x2)` on the 1-D views, where x1 / x2 are the wealth-axis coordinates
# of y_in / y_out respectively. Either x argument may be a shared 1-D
# vector (the canonical wgrid) or a per-cell N-dim array (the
# materialised wealth_post). `wdim` is the position of the wealth axis.
function _along_wealth(y_out::AbstractArray{T,N},
                       y_in::AbstractArray{T,N},
                       x_for_y_in::AbstractArray,
                       x_for_y_out::AbstractArray,
                       wdim::Int,
                       op) where {T, N}
    dims = size(y_in)
    other = ntuple(i -> i == wdim ? 1 : dims[i], N)
    if wdim == 1
        # No permutation needed — wealth axis is already leading.
        for other_ci in CartesianIndices(other)
            in_view  = view(y_in,  :, other_ci.I[2:end]...)
            out_view = view(y_out, :, other_ci.I[2:end]...)
            x_in_view  = _slice_x_along_wealth(x_for_y_in,  other_ci, wdim, N)
            x_out_view = _slice_x_along_wealth(x_for_y_out, other_ci, wdim, N)
            op(out_view, in_view, x_in_view, x_out_view)
        end
    else
        # Bring wealth dim to front, operate, permute back.
        perm     = _bring_dim_first(N, wdim)
        inv_perm = invperm(perm)
        y_in_p   = permutedims(y_in,  perm)
        y_out_p  = similar(y_in_p)
        x_in_p   = ndims(x_for_y_in)  == N ? permutedims(x_for_y_in,  perm) : x_for_y_in
        x_out_p  = ndims(x_for_y_out) == N ? permutedims(x_for_y_out, perm) : x_for_y_out
        permdims = ntuple(i -> dims[perm[i]], N)
        other_p  = ntuple(i -> i == 1 ? 1 : permdims[i], N)
        for other_ci in CartesianIndices(other_p)
            in_view  = view(y_in_p,  :, other_ci.I[2:end]...)
            out_view = view(y_out_p, :, other_ci.I[2:end]...)
            x_in_view  = _slice_x_along_wealth(x_in_p,  other_ci, 1, N)
            x_out_view = _slice_x_along_wealth(x_out_p, other_ci, 1, N)
            op(out_view, in_view, x_in_view, x_out_view)
        end
        y_out .= permutedims(y_out_p, inv_perm)
    end
    return y_out
end

# Slice a wealth-axis coordinate array (either a shared 1-D vector or a
# per-cell N-dim array) down to a 1-D view along the leading wealth
# dim. `other_ci` carries the indices into the trailing (non-wealth)
# axes; for a shared vector it's ignored.
function _slice_x_along_wealth(x::AbstractArray, other_ci::CartesianIndex,
                               wdim::Int, N::Int)
    return ndims(x) == 1 ? x : view(x, :, other_ci.I[2:end]...)
end

# Type-stable extrap value lookup from the stage's type parameter.
@inline _extrap_value(::Type{Val{:linear}}) = Val(:linear)
@inline _extrap_value(::Type{Val{:clip}})   = Val(:clip)
@inline _extrap_value(::Type{Val{-Inf}})    = Val(-Inf)
