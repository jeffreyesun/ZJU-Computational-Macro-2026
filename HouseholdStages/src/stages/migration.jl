"""
Configuration for a dedicated migration stage on a `:location`-style
categorical axis. Each cell's origin location `i` is the current value
of `location_axis`; households draw a Type-I extreme-value (Gumbel)
preference shock and choose a destination `j` with logit probability

```
P(j | i, s) = exp((-C[i, j] + a(j) + V_end[j, s]) / ε)
              / ∑_k exp((-C[i, k] + a(k) + V_end[k, s]) / ε)
```

where `C` is the migration cost matrix indexed `(origin, destination)`,
`a(j)` is an optional destination-amenity shifter (see below), `ε` is
the Gumbel scale, and `s` is the collection of non-location state
values. The value of locating at origin `i` is the log-sum-exp.

This is the same kernel as a [`LogitChoiceStage`](@ref) over the
location axis with `flow_payoff(cell, dest) = -C[cell.location_idx,
dest_idx] + a(dest_idx)`, but the API is simpler: pass the cost matrix
and `ε` as data, not as closures. The cost matrix is static on the
Spec; `ε` can be a literal or a Symbol-valued sweep key via
[`Param`](@ref).

# Destination amenity shifter

The `amenity` field carries an *additive* per-destination utility
shifter that can vary at runtime via `env`. Three forms are accepted:

  * `nothing` — no shifter; effective utility is `-C[i, j] + V_end[j, s]`
    (the original / pre-2026-05-20 behavior).
  * `AbstractVector{<:Real}` of length `n_loc` — a static shifter
    `a[j]` on each destination. Stored on the Spec; treated like
    `migration_cost` for `with_eltype`-style re-typing.
  * Closure `(destination; env) -> Real` — `destination` is the
    destination axis *value* (e.g. `:home` / `:abroad`), `env` is the
    full env NamedTuple. Materialised into the buffer's kernel on
    every `backward!`.

# Design choice (2026-05-20)

The split between static `migration_cost::Matrix` and an additive
`amenity` shifter (rather than folding both into a single
matrix-or-closure dual) keeps the hot path direct: `C[i, j]` is plain
array indexing in the triply-nested loop, and the additive shifter is
materialised into a length-`n_loc` vector once per backward pass. The
L06 Berry-contraction use case is precisely this additive form.

Pure data — no per-call buffers on the Spec.
"""
struct MigrationStageSpec{Cmat<:AbstractMatrix, Amenity, T<:Real,
                          LIn<:StateLayout, LOut<:StateLayout} <: AbstractStageSpec
    location_axis  :: Symbol
    location_dim   :: Int
    migration_cost :: Cmat
    amenity        :: Amenity
    ε              :: Param{T}
    input_layout   :: LIn
    output_layout  :: LOut
    element_type   :: Type{T}
end

"""
    MigrationStageSpec(layout; location_axis=:location, migration_cost,
                       amenity=nothing, ε, element_type=nothing)

Build the Spec for a [`MigrationStage`](@ref). The cost matrix must be
`(n_loc, n_loc)` for the named location axis. `amenity` is either
`nothing`, an `AbstractVector{<:Real}` of length `n_loc`, or a closure
`(destination; env) -> Real`. `ε` is a [`Param`](@ref) or a raw number;
`element_type` defaults to the eltype derived from `ε`'s value (or
`Float64` for a Symbol-valued sweep key).
"""
function MigrationStageSpec(layout::StateLayout;
                            location_axis::Symbol = :location,
                            migration_cost::AbstractMatrix,
                            amenity = nothing,
                            ε,
                            element_type::Union{Type, Nothing} = nothing)
    location_dim = axis_position(layout, location_axis)
    n_loc = axissize(layout.axes[location_dim])
    size(migration_cost) == (n_loc, n_loc) ||
        error("MigrationStageSpec: cost matrix has shape $(size(migration_cost)), " *
              "expected ($n_loc, $n_loc) for axis :$location_axis of size $n_loc")
    amenity_field = _wrap_migration_amenity(amenity, n_loc)
    ε_param = ε isa Param ? ε : Param(Float64(ε))
    T_default = let v = ε_param.val
        v isa Symbol ? Float64 : (typeof(v) <: Real ? typeof(v) : Float64)
    end
    T = @something element_type T_default
    return MigrationStageSpec{typeof(migration_cost), typeof(amenity_field), T,
                              typeof(layout), typeof(layout)}(
        location_axis, location_dim, migration_cost, amenity_field, ε_param,
        layout, layout, T,
    )
end

"""CLAUDE
Normalise the `amenity` argument into a stored field.

  * `nothing` passes through unchanged (no shifter).
  * `AbstractVector{<:Real}` passes through after a length check.
  * Anything else is treated as a closure `(destination; env) -> Real`
    and stored as-is (functions broadcast as scalars by default).
"""
_wrap_migration_amenity(::Nothing, ::Int) = nothing
function _wrap_migration_amenity(v::AbstractVector{<:Real}, n_loc::Int)
    length(v) == n_loc ||
        error("MigrationStageSpec: amenity vector has length $(length(v)), " *
              "expected $n_loc (one per destination location)")
    return v
end
_wrap_migration_amenity(f, ::Int) = f

"""
Per-call buffer for a migration stage. The kernel is a NamedTuple
`(; choice_prob, amenity_values)` where `choice_prob` is the per-cell
destination-probability tensor of shape `(layout_size..., n_loc)` and
`amenity_values` is the materialised length-`n_loc` shifter vector
used in this backward pass (an aliased view of the Spec's vector for
static forms; a freshly materialised vector for the closure form;
`nothing` when the Spec carries no shifter). No scratch.
"""
struct MigrationStageBuffer{T<:Real, N, AV<:AbstractArray{T,N},
                            Kernel} <: AbstractStageBuffer
    kernel  :: Kernel
    scratch :: Nothing
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A migration stage on a `:location`-style categorical axis. Construct
via `MigrationStage(layout; location_axis=:location, migration_cost,
amenity=nothing, ε)`. Composes via `∘` and `×`.
"""
struct MigrationStage{Spec<:MigrationStageSpec,
                      Buffer<:MigrationStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function MigrationStage(layout::StateLayout;
                        location_axis::Symbol = :location,
                        migration_cost::AbstractMatrix,
                        amenity = nothing,
                        ε,
                        element_type::Union{Type, Nothing} = nothing,
                        V_start::Union{Nothing, AbstractArray} = nothing,
                        Λ_end::Union{Nothing, AbstractArray}  = nothing)
    spec = MigrationStageSpec(layout; location_axis, migration_cost,
                              amenity, ε, element_type)
    return MigrationStage(spec, allocate(spec, spec.element_type;
                                         V_start, Λ_end))
end

MigrationStage(spec::MigrationStageSpec) = MigrationStage(spec, allocate(spec))
bundle(spec::MigrationStageSpec)         = MigrationStage(spec)

static_env_deps(::Type{<:MigrationStageSpec}) = NamedTuple()

# Allocate #
#----------#
# The amenity_values slot in the kernel is the destination-shifter
# vector used in the most recent backward!. Its provenance depends on
# the Spec's `amenity` field:
#
#   * nothing                  -> kernel.amenity_values = nothing
#   * AbstractVector{<:Real}   -> kernel.amenity_values aliases the Spec vector
#   * closure (dest; env)→Real -> kernel.amenity_values is a fresh Vector{T}
#
# The closure form needs a fresh buffer because backward! writes into
# it each call; the static-vector form is read-only at runtime so an
# alias is safe and avoids a copy.

function allocate(spec::MigrationStageSpec{Cmat, Amenity},
                  ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing
                  ) where {Cmat, Amenity, T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    dims  = layout_size(spec.input_layout)
    n_loc = axissize(spec.input_layout.axes[spec.location_dim])
    choice_prob   = zeros(T, dims..., n_loc)
    amenity_values = _alloc_amenity_values(spec.amenity, n_loc, T)
    kernel = (; choice_prob, amenity_values)
    return MigrationStageBuffer{T, ndims(Vs), typeof(Vs), typeof(kernel)}(
        kernel, nothing, Vs, Λe, CacheState(),
    )
end

"""CLAUDE
Allocate (or reuse) the per-call amenity-values buffer used by
`backward!`. Static-vector amenities are aliased directly so the
kernel doesn't carry a redundant copy; closures get a fresh vector
that backward! overwrites every call; `nothing` returns `nothing`.
"""
_alloc_amenity_values(::Nothing, ::Int, ::Type) = nothing
_alloc_amenity_values(v::AbstractVector{<:Real}, ::Int, ::Type) = v
_alloc_amenity_values(::Any, n_loc::Int, ::Type{T}) where {T} = zeros(T, n_loc)

# Backward #
#----------#
# V_pre[..., i, ...] = ε log Σ_j exp((-C[i, j] + a[j] + V_post[..., j, ...]) / ε)
# with `i` and `j` running over the location-axis positions and `...`
# over the non-location state. The amenity vector `a` is materialised
# from the Spec's `amenity` field (vector pass-through or closure
# evaluation) once per call.

function backward!(spec::MigrationStageSpec, V_end, env,
                   buffer::MigrationStageBuffer)
    (; input_layout, location_dim, migration_cost) = spec
    V_start = buffer.V_start
    prob    = buffer.kernel.choice_prob
    ε       = resolve(spec.ε, env)
    n_loc   = axissize(input_layout.axes[location_dim])
    dims    = layout_size(input_layout)
    T       = eltype(V_start)
    C       = migration_cost
    a       = _resolve_amenity_values!(buffer.kernel.amenity_values,
                                       spec.amenity, input_layout,
                                       location_dim, env)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        i_loc   = in_idxs[location_dim]

        # Pass 1: max for numerical stability.
        max_u = typemin(T)
        for j in 1:n_loc
            out_idxs = Base.setindex(in_idxs, j, location_dim)
            u = -C[i_loc, j] + _amenity_at(a, j) + V_end[CartesianIndex(out_idxs)]
            u > max_u && (max_u = u)
        end

        # Pass 2: unnormalised weights.
        denom = zero(T)
        for j in 1:n_loc
            out_idxs = Base.setindex(in_idxs, j, location_dim)
            u = -C[i_loc, j] + _amenity_at(a, j) + V_end[CartesianIndex(out_idxs)]
            w = exp((u - max_u) / ε)
            prob[in_idxs..., j] = w
            denom += w
        end

        # Pass 3: normalise.
        for j in 1:n_loc
            prob[in_idxs..., j] /= denom
        end
        V_start[ci] = max_u + ε * log(denom)
    end
    _seat_cache!(buffer, V_end, env)
    return V_start
end

"""CLAUDE
Materialise the destination-amenity vector for the current backward
pass. Static-vector form is a no-op (returns the alias); closure form
broadcasts the closure across destination axis values into the
pre-allocated buffer; `nothing` returns `nothing` (the inner loop then
contributes zero via `_amenity_at`).
"""
_resolve_amenity_values!(::Nothing, ::Nothing, ::StateLayout, ::Int, _env) = nothing
_resolve_amenity_values!(buf::AbstractVector, ::AbstractVector{<:Real},
                         ::StateLayout, ::Int, _env) = buf
function _resolve_amenity_values!(buf::AbstractVector, f, layout::StateLayout,
                                  location_dim::Int, env)
    dests = axisvalues(layout.axes[location_dim])
    @inbounds for j in eachindex(buf)
        buf[j] = f(dests[j]; env)
    end
    return buf
end

# Per-destination amenity lookup at index `j`. `nothing` contributes
# nothing (zero); a vector indexes into the materialised values.
@inline _amenity_at(::Nothing, ::Int) = false  # zero under +, type-promotion-friendly
@inline _amenity_at(a::AbstractVector, j::Int) = @inbounds a[j]

# Forward #
#---------#
# Λ_post[j, s] = Σ_i P(j | i, s) · Λ_pre[i, s]

function forward!(spec::MigrationStageSpec, Λ_start,
                  buffer::MigrationStageBuffer)
    (; input_layout, location_dim) = spec
    Λ_end = buffer.Λ_end
    prob  = buffer.kernel.choice_prob
    n_loc = axissize(input_layout.axes[location_dim])
    dims  = layout_size(input_layout)
    T     = eltype(Λ_end)

    fill!(Λ_end, zero(T))
    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        mass    = Λ_start[ci]
        iszero(mass) && continue
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, location_dim)
            Λ_end[CartesianIndex(out_idxs)] += mass * p
        end
    end
    return Λ_end
end
