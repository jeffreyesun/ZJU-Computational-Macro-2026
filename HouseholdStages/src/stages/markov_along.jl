using LinearAlgebra: mul!

"""
Markov transition along one named axis. K is the transition matrix
itself (V/θ-independent). Backward applies `K^T`; forward applies `K`.
"""
struct MarkovStageSpec{M<:AbstractMatrix, T<:Real} <: AbstractStageSpec
    transition :: M
    axis       :: Symbol
end

function MarkovStageSpec(; axis::Symbol, transition::AbstractMatrix)
    @assert size(transition, 1) == size(transition, 2)
    T = eltype(transition)
    return MarkovStageSpec{typeof(transition), T}(transition, axis)
end

"Scratch: two permuted-axis buffers used when `axis_dim != 1`."
struct MarkovScratch{A<:AbstractArray}
    perm_in  :: A
    perm_out :: A
end

function allocate_scratch(spec::MarkovStageSpec, ::Type{T}, layout::StateLayout) where {T}
    dims     = layout_size(layout)
    axis_dim = axis_position(layout, spec.axis)
    perm     = _bring_dim_first(length(dims), axis_dim)
    permdims = ntuple(i -> dims[perm[i]], length(dims))
    return MarkovScratch(zeros(T, permdims), zeros(T, permdims))
end

# Internals #
#-----------#

function _permute_mul!(perm_out, M, perm_in)
    n         = size(perm_in, 1)
    rest      = div(length(perm_in), n)
    pflat_in  = reshape(perm_in, n, rest)
    pflat_out = reshape(perm_out, n, rest)
    mul!(pflat_out, M, pflat_in)
    return perm_out
end

# Apply M along `axis` of `src` into `dest` via permuted-layout scratch.
# Fast path: when `axis` is already first, `permute_to_first!` returns
# `src` unchanged and we mul! straight into `dest`. Slow path: round-trip
# through scratch.
function _markov_apply!(dest, src, M, layout::StateLayout, axis::Symbol, perm_in, perm_out)
    src_p = permute_to_first!(perm_in, src, layout, axis)
    if src_p === src
        return _permute_mul!(dest, M, src)
    end
    _permute_mul!(perm_out, M, src_p)
    perm = _bring_dim_first(ndims(src), axis_dim(layout, axis))
    return permutedims!(dest, perm_out, invperm(perm))
end

# Backward / forward #
#--------------------#

function backward!(buffer, spec::MarkovStageSpec, V_end, env)
    _markov_apply!(buffer.V_start, V_end, spec.transition,
                   buffer.input_layout, spec.axis,
                   buffer.scratch.perm_in, buffer.scratch.perm_out)
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

function forward!(buffer, spec::MarkovStageSpec, Λ_start)
    _markov_apply!(buffer.Λ_end, Λ_start, spec.transition',
                   buffer.input_layout, spec.axis,
                   buffer.scratch.perm_in, buffer.scratch.perm_out)
    return buffer.Λ_end
end

# Wrapper #
#---------#

@definestage MarkovStage MarkovStageSpec scratch=MarkovScratch
