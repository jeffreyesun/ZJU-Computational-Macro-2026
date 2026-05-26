"""
Deterministic wealth-change stage. Each cell's wealth transitions
according to `wealth_post(cell; env) -> Real`. Backward interpolates
`V_end` along the wealth axis at the per-cell post-wealth; forward
redistributes mass via shares. `extrap` ∈ `(:linear, :clip, -Inf)`
controls the backward interpolation policy for the underflow region.
"""
struct WealthChangeStageSpec{F, Extrap} <: AbstractStageSpec
    wealth_post :: F
    wealth_axis :: Symbol
end

function WealthChangeStageSpec(; wealth_post, wealth_axis::Symbol=:wealth, extrap=:linear)
    @assert extrap in (:linear, :clip, -Inf)
    return WealthChangeStageSpec{typeof(wealth_post), Val{extrap}}(wealth_post, wealth_axis)
end

"Kernel: per-cell post-wealth array used by both backward (interp) and forward (push)."
struct WealthChangeKernel{P<:AbstractArray}
    wealth_post :: P
end

"Scratch: the memoised `cell_array(layout)` so the closure broadcast doesn't reallocate."
struct WealthChangeScratch{C}
    cells :: C
end

allocate_kernel(::WealthChangeStageSpec, ::Type{T}, layout::StateLayout) where {T} =
    WealthChangeKernel(zeros(T, layout_size(layout)))

allocate_scratch(::WealthChangeStageSpec, ::Type, layout::StateLayout) =
    WealthChangeScratch(cell_array(layout))

# Backward / forward #
#--------------------#

function backward!(buffer, spec::WealthChangeStageSpec{F, Extrap},
                   V_end, env) where {F, Extrap}
    layout     = buffer.input_layout
    wgrid      = axisvalues(layout.axes[axis_dim(layout, spec.wealth_axis)])
    wpost      = buffer.kernel.wealth_post
    V_start    = buffer.V_start

    wpost .= spec.wealth_post.(buffer.scratch.cells; env)
    extrap_val = _extrap_value(Extrap)
    _along_wealth(V_start, V_end, wgrid, wpost, layout, spec.wealth_axis,
                  (y2, y1, x1, x2) -> reinterpolate!(y2, y1, x1, x2, extrap_val))
    _seat_cache!(buffer, V_end, env)
    return V_start
end

function forward!(buffer, spec::WealthChangeStageSpec, Λ_start)
    layout     = buffer.input_layout
    wgrid      = axisvalues(layout.axes[axis_dim(layout, spec.wealth_axis)])
    wpost      = buffer.kernel.wealth_post
    Λ_end      = buffer.Λ_end
    _along_wealth(Λ_end, Λ_start, wpost, wgrid, layout, spec.wealth_axis,
                  (y2, y1, x1, x2) -> convert_distribution!(y2, y1, x1, x2, Val(:share)))
    return Λ_end
end

# Internals #
#-----------#

# Iterate the wealth axis at every (other-axis) slice and call `op(y2, y1, x1, x2)`
# on the 1-D views. Either x argument may be a shared 1-D vector (the canonical
# wgrid) or a per-cell N-dim array (the materialised wealth_post). `slices_over`
# yields the per-(other-axis) view along wealth; N-D `x_*` arrays are sliced the
# same way, 1-D `x_*` vectors pass through unchanged.
function _along_wealth(y_out::AbstractArray{T,N},
                       y_in::AbstractArray{T,N},
                       x_for_y_in::AbstractArray,
                       x_for_y_out::AbstractArray,
                       layout::StateLayout, wealth_axis::Symbol, op) where {T, N}
    y_in_slices  = slices_over(y_in,  layout, wealth_axis)
    y_out_slices = slices_over(y_out, layout, wealth_axis)
    xin_is_nd    = ndims(x_for_y_in)  == N
    xout_is_nd   = ndims(x_for_y_out) == N
    x_in_slices  = xin_is_nd  ? slices_over(x_for_y_in,  layout, wealth_axis) : x_for_y_in
    x_out_slices = xout_is_nd ? slices_over(x_for_y_out, layout, wealth_axis) : x_for_y_out

    for ci in CartesianIndices(y_in_slices)
        x_in_view  = xin_is_nd  ? x_in_slices[ci]  : x_in_slices
        x_out_view = xout_is_nd ? x_out_slices[ci] : x_out_slices
        op(y_out_slices[ci], y_in_slices[ci], x_in_view, x_out_view)
    end
    return y_out
end

_extrap_value(::Type{Val{:linear}}) = Val(:linear)
_extrap_value(::Type{Val{:clip}})   = Val(:clip)
_extrap_value(::Type{Val{-Inf}})    = Val(-Inf)

# Wrapper #
#---------#

@definestage WealthChangeStage WealthChangeStageSpec kernel=WealthChangeKernel scratch=WealthChangeScratch
