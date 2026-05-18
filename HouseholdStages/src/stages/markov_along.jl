using LinearAlgebra: mul!

"""
A stage that applies a Markov transition along one named axis of the
state layout. The K-operator is the transition matrix itself
(V/θ-independent); backward applies `K^T` along the axis, forward
applies `K`. Construct via `MarkovAlong(layout; axis, transition)`. The
buffer eltype defaults to `eltype(transition)` and is parametric in `T`
to support AD via [`with_eltype`](@ref).
"""
struct MarkovAlong{M<:AbstractMatrix, T<:Real, N,
                   LIn<:StateLayout, LOut<:StateLayout,
                   AV<:AbstractArray{T,N}} <: AbstractStage
    transition    :: M
    axis          :: Symbol
    axis_dim      :: Int
    input_layout  :: LIn
    output_layout :: LOut
    V_start       :: AV
    Λ_end         :: AV
end

"""
    MarkovAlong(layout; axis, transition, element_type=eltype(transition)) -> MarkovAlong

Build a Markov stage on `layout` over the axis named `axis`. The
transition matrix is square and must match the axis's size.

`element_type` controls the buffer eltype (default = `eltype(transition)`).
Set explicitly (e.g., to `ForwardDiff.Dual{…}`) when rebuilding the stage
for AD — see [`lift_jacobian`](@ref).
"""
function MarkovAlong(layout::StateLayout;
                     axis::Symbol,
                     transition::AbstractMatrix,
                     element_type::Type{T} = eltype(transition),
                     V_start::Union{Nothing, AbstractArray} = nothing,
                     Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    axis_dim = axis_position(layout, axis)
    n_axis   = axissize(layout.axes[axis_dim])
    @assert size(transition, 1) == size(transition, 2) "transition must be square"
    @assert size(transition, 1) == n_axis "transition size $(size(transition,1)) must match axis :$axis size $n_axis"

    dims = layout_size(layout)
    N    = length(dims)
    Vs   = V_start === nothing ? zeros(T, dims) : V_start
    Λe   = Λ_end   === nothing ? zeros(T, dims) : Λ_end
    @assert typeof(Vs) === typeof(Λe) "MarkovAlong: V_start and Λ_end must have the same concrete array type; got $(typeof(Vs)) and $(typeof(Λe))"
    return MarkovAlong{typeof(transition), T, N, typeof(layout), typeof(layout), typeof(Vs)}(
        transition, axis, axis_dim, layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:MarkovAlong}) = NamedTuple()

# Allocate #
#----------#

# Kernel = nothing: K is the transition matrix, V/θ-independent and on
# the stage struct itself. Scratch holds two permuted-axis buffers for
# non-trivial axis_dim.
function allocate(stage::MarkovAlong{M,T,N}, ::Type{T2} = T) where {M,T,N,T2}
    dims     = size(stage.V_start)
    perm     = _bring_dim_first(N, stage.axis_dim)
    permdims = ntuple(i -> dims[perm[i]], N)
    perm_in  = zeros(T2, permdims)
    perm_out = zeros(T2, permdims)
    return (nothing, (perm_in = perm_in, perm_out = perm_out))
end

# Backward #
#----------#

function backward!(stage::MarkovAlong{M,T,N},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {M,T,N}
    _markov_apply!(stage.V_start, V_end, stage.transition,
                   stage.axis_dim, scratch.perm_in, scratch.perm_out)
    return stage.V_start
end

# Forward #
#---------#

function forward!(stage::MarkovAlong{M,T,N},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {M,T,N}
    _markov_apply!(stage.Λ_end, Λ_start, stage.transition',
                   stage.axis_dim, scratch.perm_in, scratch.perm_out)
    return stage.Λ_end
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
