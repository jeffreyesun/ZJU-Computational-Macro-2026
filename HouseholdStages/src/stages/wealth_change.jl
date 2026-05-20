"""
Configuration for a deterministic wealth-change stage. Each cell's
wealth transitions according to the user closure
`wealth_post(cell; env) -> Real`. The K-operator is "look up the value
at `wealth_post(cell; env)` along the wealth axis": backward uses
linear interpolation on `V_end`, forward uses share-based mass
redistribution. Pure data — no per-call buffers.

Other state axes pass through unchanged. The `:wealth` axis is
expected to be a [`ContinuousGrid`](@ref); other axes can be of any
kind.

Two extrapolation policies are wired up via the `extrap` kwarg for the
backward V-interpolation, encoded as a `Val`-typed type parameter:

  * `:linear` — extend the leftmost slope into the underflow region.
  * `:clip`   — snap underflow to `V_end[1, ...]`.
  * `-Inf`    — mark underflow cells as unreachable (useful for
    enforcing borrowing constraints).

Underflow / overflow on the forward Λ-push are handled by
[`convert_distribution!`](@ref): mass landing below `wgrid[1]`
accumulates in `wgrid[1]`, mass landing past `wgrid[end]` accumulates in
`wgrid[end]`.

`env` is passed plainly to the user closure (no `Ref` wrapper); access
`env.r`, `env.w` etc. directly.
"""
struct WealthChangeStageSpec{F, T<:Real, L<:StateLayout, Extrap} <: AbstractStageSpec
    wealth_post   :: F
    wealth_axis   :: Symbol
    wealth_dim    :: Int
    input_layout  :: L
    output_layout :: L
    element_type  :: Type{T}
end

"""
    WealthChangeStageSpec(layout; wealth_post, wealth_axis=:wealth,
                          extrap=:linear, element_type=Float64)

Build the Spec for a [`WealthChangeStage`](@ref). `extrap` is
`:linear` / `:clip` / `-Inf`.
"""
function WealthChangeStageSpec(layout::StateLayout;
                               wealth_post,
                               wealth_axis::Symbol = :wealth,
                               extrap = :linear,
                               element_type::Type{T} = Float64) where {T<:Real}
    extrap in (:linear, :clip, -Inf) ||
        error("WealthChangeStageSpec: extrap must be :linear, :clip, or -Inf; got $extrap")
    wealth_dim = axis_position(layout, wealth_axis)
    return WealthChangeStageSpec{typeof(wealth_post), T, typeof(layout),
                                 Val(extrap) |> typeof}(
        wealth_post, wealth_axis, wealth_dim, layout, layout, element_type,
    )
end

"""
Per-call buffer for a wealth-change stage. The kernel is a NamedTuple
`(; wealth_post)` holding the materialised per-cell post-wealth array.
Scratch is `(; cells)`: the memoised `cell_array(layout)` so the
per-backward closure broadcast doesn't reallocate.
"""
struct WealthChangeStageBuffer{T<:Real, N, AV<:AbstractArray{T,N},
                               Kernel, Scratch} <: AbstractStageBuffer
    kernel  :: Kernel
    scratch :: Scratch
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A deterministic wealth-change stage. Construct via
`WealthChangeStage(layout; wealth_post, wealth_axis=:wealth,
extrap=:linear)`. Composes via `∘` and `×`. The `Extrap` policy is
carried as a `Val`-typed type parameter on the Spec for type-stable
dispatch in the backward interpolation.
"""
struct WealthChangeStage{Spec<:WealthChangeStageSpec,
                         Buffer<:WealthChangeStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function WealthChangeStage(layout::StateLayout;
                           wealth_post,
                           wealth_axis::Symbol = :wealth,
                           extrap = :linear,
                           element_type::Type{T} = Float64,
                           V_start::Union{Nothing, AbstractArray} = nothing,
                           Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    spec = WealthChangeStageSpec(layout; wealth_post, wealth_axis, extrap, element_type)
    return WealthChangeStage(spec, allocate(spec, T; V_start, Λ_end))
end

WealthChangeStage(spec::WealthChangeStageSpec) = WealthChangeStage(spec, allocate(spec))
bundle(spec::WealthChangeStageSpec)            = WealthChangeStage(spec)

static_env_deps(::Type{<:WealthChangeStageSpec}) = NamedTuple()

# Allocate #
#----------#
# Kernel holds the materialised `wealth_post` array (same shape as the
# state layout). Scratch holds the memoised `cell_array(layout)` so the
# closure broadcast doesn't reallocate every backward pass.
function allocate(spec::WealthChangeStageSpec, ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    dims        = layout_size(spec.input_layout)
    wealth_post = zeros(T, dims)
    kernel      = (; wealth_post)
    cells       = cell_array(spec.input_layout)
    scratch     = (; cells)
    return WealthChangeStageBuffer{T, ndims(Vs), typeof(Vs), typeof(kernel), typeof(scratch)}(
        kernel, scratch, Vs, Λe, CacheState(),
    )
end

# Backward #
#----------#
# V_pre(cell) = V_post evaluated at wealth_post(cell; env), interpolated
# along the wealth axis using the chosen extrap policy.

function backward!(spec::WealthChangeStageSpec{F,T,L,Extrap},
                   V_end, env, buffer::WealthChangeStageBuffer) where {F,T,L,Extrap}
    (; kernel, scratch, V_start) = buffer
    wgrid = axisvalues(spec.input_layout.axes[spec.wealth_dim])
    wpost = kernel.wealth_post

    # Evaluate the post-stage wealth at every cell. The kwarg `env` is
    # captured (not broadcast), so the closure sees env unwrapped.
    wpost .= spec.wealth_post.(scratch.cells; env)

    # V_end lives on wgrid; we want V_start at the per-cell query points wpost.
    extrap_val = _extrap_value(Extrap)
    _along_wealth(V_start, V_end, wgrid, wpost, spec.wealth_dim,
                  (y2, y1, x1, x2) -> reinterpolate!(y2, y1, x1, x2, extrap_val))
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward #
#---------#
# Λ_post on wgrid = convert_distribution(Λ_pre with source positions
# wpost → wgrid), via share-based redistribution.

function forward!(spec::WealthChangeStageSpec, Λ_start, buffer::WealthChangeStageBuffer)
    (; kernel, Λ_end) = buffer
    wgrid = axisvalues(spec.input_layout.axes[spec.wealth_dim])
    wpost = kernel.wealth_post
    # Λ_start has source positions wpost (per cell); Λ_end lives on wgrid.
    _along_wealth(Λ_end, Λ_start, wpost, wgrid, spec.wealth_dim,
                  (y2, y1, x1, x2) -> convert_distribution!(y2, y1, x1, x2, Val(:share)))
    return Λ_end
end

# Internals #
#-----------#

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

# Type-stable extrap value lookup from the spec's type parameter.
@inline _extrap_value(::Type{Val{:linear}}) = Val(:linear)
@inline _extrap_value(::Type{Val{:clip}})   = Val(:clip)
@inline _extrap_value(::Type{Val{-Inf}})    = Val(-Inf)
