using LinearAlgebra: mul!

"""
Configuration for a Markov-along-axis stage. The K-operator is the
transition matrix itself (V/θ-independent); backward applies `K^T`,
forward applies `K`. Pure data — no per-call buffers.
"""
struct MarkovStageSpec{M<:AbstractMatrix, T<:Real,
                       LIn<:StateLayout, LOut<:StateLayout} <: AbstractStageSpec
    transition    :: M
    axis          :: Symbol
    axis_dim      :: Int
    input_layout  :: LIn
    output_layout :: LOut
    element_type  :: Type{T}
end

"""
    MarkovStageSpec(layout; axis, transition, element_type=eltype(transition))

Build the Spec for a Markov stage. The transition matrix must be
square and match the named axis's size.
"""
function MarkovStageSpec(layout::StateLayout;
                         axis::Symbol,
                         transition::AbstractMatrix,
                         element_type::Type{T} = eltype(transition)) where {T<:Real}
    axis_dim = axis_position(layout, axis)
    n_axis   = axissize(layout.axes[axis_dim])
    @assert size(transition, 1) == size(transition, 2) "transition must be square"
    @assert size(transition, 1) == n_axis "transition size $(size(transition,1)) must match axis :$axis size $n_axis"
    return MarkovStageSpec{typeof(transition), T, typeof(layout), typeof(layout)}(
        transition, axis, axis_dim, layout, layout, element_type,
    )
end

"""
Per-call buffer for a Markov stage. Kernel = `nothing` (K is the
transition matrix, V/θ-independent and on the Spec). Scratch holds
two permuted-axis buffers used when `axis_dim != 1`.
"""
struct MarkovStageBuffer{T<:Real, N, AV<:AbstractArray{T,N},
                         Scratch} <: AbstractStageBuffer
    kernel  :: Nothing
    scratch :: Scratch
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A Markov transition along one named axis. Construct via
`MarkovStage(layout; axis, transition)`. Composes via `∘` and `×`.
"""
struct MarkovStage{Spec<:MarkovStageSpec, Buffer<:MarkovStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function MarkovStage(layout::StateLayout;
                     axis::Symbol,
                     transition::AbstractMatrix,
                     element_type::Type{T} = eltype(transition),
                     V_start::Union{Nothing, AbstractArray} = nothing,
                     Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    spec = MarkovStageSpec(layout; axis, transition, element_type)
    return MarkovStage(spec, allocate(spec, T; V_start, Λ_end))
end

MarkovStage(spec::MarkovStageSpec) = MarkovStage(spec, allocate(spec))
bundle(spec::MarkovStageSpec)      = MarkovStage(spec)

static_env_deps(::Type{<:MarkovStageSpec}) = NamedTuple()

# Allocate #
#----------#

function allocate(spec::MarkovStageSpec{M,T1,LIn,LOut},
                  ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {M,T1,LIn,LOut,T}
    dims     = layout_size(spec.input_layout)
    N        = length(dims)
    perm     = _bring_dim_first(N, spec.axis_dim)
    permdims = ntuple(i -> dims[perm[i]], N)
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    scratch  = (; perm_in = zeros(T, permdims), perm_out = zeros(T, permdims))
    return MarkovStageBuffer{T, N, typeof(Vs), typeof(scratch)}(
        nothing, scratch, Vs, Λe, CacheState(),
    )
end

# Backward #
#----------#

function backward!(spec::MarkovStageSpec, V_end, env, buffer::MarkovStageBuffer)
    _markov_apply!(buffer.V_start, V_end, spec.transition,
                   spec.axis_dim, buffer.scratch.perm_in, buffer.scratch.perm_out)
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

# Forward #
#---------#

function forward!(spec::MarkovStageSpec, Λ_start, buffer::MarkovStageBuffer)
    _markov_apply!(buffer.Λ_end, Λ_start, spec.transition',
                   spec.axis_dim, buffer.scratch.perm_in, buffer.scratch.perm_out)
    return buffer.Λ_end
end

# Internals #
#-----------#

# Apply M along `dim` of `src` into `dest`. Both `dest` and `src` are in
# *original* shape; `perm_in` and `perm_out` are scratch buffers in the
# permuted shape (axis `dim` brought to front). `M` is square.
function _markov_apply!(dest::AbstractArray{T,N}, src::AbstractArray{T,N}, M,
                        dim::Int, perm_in::AbstractArray{T,N},
                        perm_out::AbstractArray{T,N}) where {T,N}
    @assert size(dest) == size(src)
    @assert size(M, 1) == size(src, dim) == size(M, 2)

    perm = _bring_dim_first(N, dim)

    if dim == 1
        n = size(src, 1)
        rest = div(length(src), n)
        sflat = reshape(src, n, rest)
        dflat = reshape(dest, n, rest)
        mul!(dflat, M, sflat)
        return dest
    end

    permutedims!(perm_in, src, perm)
    n = size(perm_in, 1)
    rest = div(length(perm_in), n)
    pflat_in  = reshape(perm_in, n, rest)
    pflat_out = reshape(perm_out, n, rest)
    mul!(pflat_out, M, pflat_in)

    inv_perm = invperm(perm)
    permutedims!(dest, perm_out, inv_perm)
    return dest
end

function _bring_dim_first(N::Int, dim::Int)
    @assert 1 <= dim <= N
    return ntuple(i -> i == 1 ? dim : (i <= dim ? i - 1 : i), N)
end
