using LinearAlgebra: mul!, transpose

"""
Migration stage on a categorical location axis. Households draw a
Type-I extreme-value (Gumbel) preference shock and choose a
destination via logit:

```
P(j | i, s) = exp((-C[i, j] + a[j] + V_end[j, s]) / ε)
              / Σ_k exp((-C[i, k] + a[k] + V_end[k, s]) / ε)
```

`migration_cost` is `(n_loc, n_loc)` indexed `(origin, destination)`.
`amenity` is an `AbstractVector{<:Real}` of length `n_loc` —
the additive destination shifter (defaults to a zero vector). For
env-dependent shifters, hold a reference to the vector and mutate
it between solves; `backward!` reads the current contents each
call. `ε` is the Gumbel scale — a literal value or `FromEnv(:key)`.
"""
struct MigrationStageSpec{Cmat<:AbstractMatrix, V<:AbstractVector, T} <: AbstractStageSpec
    location_axis  :: Symbol
    n_loc          :: Int
    migration_cost :: Cmat
    amenity        :: V
    ε              :: T
end

function MigrationStageSpec(; location_axis::Symbol=:location,
                            ε,
                            migration_cost::AbstractMatrix,
                            amenity::AbstractVector{<:Real}=zero(first(eachcol(migration_cost))))
    @assert size(migration_cost, 1) == size(migration_cost, 2) == length(amenity)
    return MigrationStageSpec(location_axis, length(amenity), migration_cost, amenity, ε)
end

# Kernel #
#--------#

"Kernel: per-(origin, destination, non-loc-state) choice-probability tensor of shape `(layout_size..., n_loc)`."
struct MigrationKernel{P<:AbstractArray}
    choice_prob :: P
end

function allocate_kernel(spec::MigrationStageSpec, ::Type{T}, layout::StateLayout) where {T}
    n_loc = length(spec.amenity)
    return MigrationKernel(zeros(T, (layout_size(layout)..., n_loc)))
end

"Scratch: contiguous `(n_loc, n_loc)` payoff buffer so the per-slice broadcast doesn't allocate."
struct MigrationScratch{P<:AbstractMatrix}
    payoff :: P
end

allocate_scratch(spec::MigrationStageSpec, ::Type{T}, ::StateLayout) where {T} =
    MigrationScratch(zeros(T, spec.n_loc, spec.n_loc))

# Backward / forward #
#--------------------#
# V_pre[..., i, ...] = ε log Σ_j exp((-C[i, j] + a[j] + V_post[..., j, ...]) / ε)
# Each non-location slice is an `(n_loc, n_loc)` problem — softmax along
# destinations for each origin. Iterate the slices; bulk-vectorize within.

function backward!(buffer, spec::MigrationStageSpec, V_end, env)
    (;ε, migration_cost, amenity, V_start) = resolve(buffer, spec, env)
    layout = buffer.input_layout
    payoff = buffer.scratch.payoff

    V_end_slices   = slices_over(V_end,   layout, spec.location_axis)
    V_start_slices = slices_over(V_start, layout, spec.location_axis)
    # `choice_prob` carries an extra trailing destination axis (not in the
    # layout), so it can't go through `slices_over`; we fix only the
    # layout's non-location dims and let the destination axis stay free
    # alongside the origin axis inside each slice.
    other_layout_dims = ntuple(i -> i < axis_dim(layout, spec.location_axis) ?
                                    i : i + 1,
                               length(layout) - 1)
    prob_slices = eachslice(buffer.kernel.choice_prob; dims=other_layout_dims)
    @inbounds for ci in CartesianIndices(V_end_slices)
        V_end_slice   = V_end_slices[ci]
        V_start_slice = V_start_slices[ci]
        prob_slice    = prob_slices[ci]
        # U[i, j] = -C[i, j] + a[j] + V_end[j], assembled in the contiguous
        # scratch so the broadcast stays on the BLAS-fast path.
        @. payoff = -migration_cost + amenity' + V_end_slice'
        _softmax_and_lse_along_last!(V_start_slice, payoff, ε)
        copyto!(prob_slice, payoff)
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

function forward!(buffer, spec::MigrationStageSpec, Λ_start)
    (;Λ_end) = resolve(buffer, spec)
    layout = buffer.input_layout

    Λ_start_slices = slices_over(Λ_start, layout, spec.location_axis)
    Λ_end_slices   = slices_over(Λ_end,   layout, spec.location_axis)
    other_layout_dims = ntuple(i -> i < axis_dim(layout, spec.location_axis) ?
                                    i : i + 1,
                               length(layout) - 1)
    prob_slices = eachslice(buffer.kernel.choice_prob; dims=other_layout_dims)
    @inbounds for ci in CartesianIndices(Λ_start_slices)
        # prob_slice[i, j] is P(j | i, s); Λ_end[j] = Σ_i prob[i,j] · Λ_start[i].
        mul!(Λ_end_slices[ci], transpose(prob_slices[ci]), Λ_start_slices[ci])
    end
    return Λ_end
end

# Wrapper #
#---------#

@definestage MigrationStage MigrationStageSpec kernel=MigrationKernel scratch=MigrationScratch
