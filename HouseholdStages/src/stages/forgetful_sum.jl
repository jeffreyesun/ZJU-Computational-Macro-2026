"""
Configuration for a [`ForgetfulSumStage`](@ref): a layout-changing
stage that drops one axis of the state space, pooling households who
are otherwise identical along the dropped axis. The output layout is
`input_layout` with `forget_axis` removed.

The K-operator is the sum-along-axis operator, fully determined by the
layout difference (no V/θ dependence; kernel = `nothing`). Backward
broadcasts `V_end` along the dropped axis; forward sums `Λ_start` along
it. Pure data — no per-call buffers.
"""
struct ForgetfulSumStageSpec{T<:Real,
                             LIn<:StateLayout,
                             LOut<:StateLayout} <: AbstractStageSpec
    forget_axis   :: Symbol
    forget_dim    :: Int
    input_layout  :: LIn
    output_layout :: LOut
    element_type  :: Type{T}
end

"""
    ForgetfulSumStageSpec(layout; forget_axis, element_type=Float64)

Build the Spec for a forgetful-sum stage. The output layout is `layout`
with `forget_axis` dropped.
"""
function ForgetfulSumStageSpec(layout::StateLayout;
                               forget_axis::Symbol,
                               element_type::Type{T} = Float64) where {T<:Real}
    forget_dim    = axis_position(layout, forget_axis)
    output_layout = drop_axis(layout, forget_axis)
    return ForgetfulSumStageSpec{T, typeof(layout), typeof(output_layout)}(
        forget_axis, forget_dim, layout, output_layout, element_type,
    )
end

"""
Per-call buffer for a forgetful-sum stage. Kernel = `nothing` (K is the
sum-along-axis operator, fully determined by the layout difference).
Scratch = `nothing` (no workspace needed). Asymmetric shape: `V_start`
is shaped by the *input* layout (input has the extra axis); `Λ_end` is
shaped by the *output* layout (the dropped axis is gone).
"""
struct ForgetfulSumStageBuffer{T<:Real, Nin, Nout,
                               AVin<:AbstractArray{T,Nin},
                               AVout<:AbstractArray{T,Nout}} <: AbstractStageBuffer
    kernel  :: Nothing
    scratch :: Nothing
    V_start :: AVin
    Λ_end   :: AVout
    cache   :: CacheState
end

"""
A layout-changing stage that drops one axis of the state space.
Construct via `ForgetfulSumStage(layout; forget_axis)`. The output
layout is `layout` with `forget_axis` removed; backward broadcasts
`V_end` along the dropped axis and forward sums `Λ_start` along it.
"""
struct ForgetfulSumStage{Spec<:ForgetfulSumStageSpec,
                         Buffer<:ForgetfulSumStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function ForgetfulSumStage(layout::StateLayout;
                           forget_axis::Symbol,
                           element_type::Type{T} = Float64,
                           V_start::Union{Nothing, AbstractArray} = nothing,
                           Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    spec = ForgetfulSumStageSpec(layout; forget_axis, element_type)
    return ForgetfulSumStage(spec, allocate(spec, T; V_start, Λ_end))
end

ForgetfulSumStage(spec::ForgetfulSumStageSpec) = ForgetfulSumStage(spec, allocate(spec))
bundle(spec::ForgetfulSumStageSpec)            = ForgetfulSumStage(spec)

static_env_deps(::Type{<:ForgetfulSumStageSpec}) = NamedTuple()

# Allocate #
#----------#

function allocate(spec::ForgetfulSumStageSpec, ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T}
    dims_in  = layout_size(spec.input_layout)
    dims_out = layout_size(spec.output_layout)
    Vs = V_start === nothing ? zeros(T, dims_in)  : V_start
    Λe = Λ_end   === nothing ? zeros(T, dims_out) : Λ_end
    return ForgetfulSumStageBuffer{T, length(dims_in), length(dims_out),
                                   typeof(Vs), typeof(Λe)}(
        nothing, nothing, Vs, Λe, CacheState(),
    )
end

# Backward: broadcast V_end along the dropped axis #
#--------------------------------------------------#

function backward!(spec::ForgetfulSumStageSpec, V_end, env,
                   buffer::ForgetfulSumStageBuffer)
    dims_out       = layout_size(spec.output_layout)
    @assert size(V_end) == dims_out
    shape          = _insert_singleton(dims_out, spec.forget_dim)
    V_end_reshaped = reshape(V_end, shape)
    buffer.V_start .= V_end_reshaped
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

# Forward: sum Λ_start along the dropped axis #
#---------------------------------------------#

function forward!(spec::ForgetfulSumStageSpec, Λ_start,
                  buffer::ForgetfulSumStageBuffer)
    dims_in  = layout_size(spec.input_layout)
    dims_out = layout_size(spec.output_layout)
    @assert size(Λ_start) == dims_in
    shape          = _insert_singleton(dims_out, spec.forget_dim)
    Λ_end_reshaped = reshape(buffer.Λ_end, shape)
    sum!(Λ_end_reshaped, Λ_start)
    return buffer.Λ_end
end

# Internals #
#-----------#

function _insert_singleton(dims::NTuple{N,Int}, pos::Int) where {N}
    @assert 1 <= pos <= N + 1
    return ntuple(i -> i < pos  ? dims[i]   :
                       i == pos ? 1         :
                                  dims[i-1], Val(N+1))
end
