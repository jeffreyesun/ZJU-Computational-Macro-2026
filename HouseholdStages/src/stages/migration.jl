"""CLAUDE
Dedicated migration stage on a `:location`-style categorical axis. Each
cell's origin location `i` is the current value of `location_axis`;
households draw a Type-I extreme-value (Gumbel) preference shock and
choose a destination `j` with logit probability

```
P(j | i, s) = exp((-C[i, j] + V_end[j, s]) / ε) / ∑_k exp((-C[i, k] + V_end[k, s]) / ε)
```

where `C` is the migration cost matrix indexed `(origin, destination)`,
`ε` is the Gumbel scale (preference-shock dispersion), and `s` is the
collection of non-location state values (wealth, income, …). The
continuation value at the destination is `V_end[j, s]`; the value of
locating at origin `i` is then the log-sum-exp.

This is the same kernel as a [`LogitChoice`](@ref) over the location
axis with `flow_payoff(cell, dest) = -C[cell.location_idx, dest_idx]`,
but the API is simpler: pass the cost matrix and `ε` as data, not as
closures. The cost matrix is static on the stage; `ε` can be a literal
or a Symbol-valued sweep key via [`Param`](@ref).

Construction:

```julia
stage = Migration(layout;
    location_axis  = :location,
    migration_cost = [0.0 0.5; 0.5 0.0],   # (n_loc × n_loc)
    ε              = 5.0,
)
```

The matrix's diagonal is the "stay" cost (typically zero); off-diagonal
entries are the "move" cost from row to column. The matrix must be
shape `(n_loc, n_loc)` where `n_loc` is the layout's location-axis
size. The matrix is stored on the struct and shared across all
non-location cells; if a per-cell cost is required, fall back to
[`LogitChoice`](@ref) with a `flow_payoff` closure.

Forward / backward / adjoints mirror [`LogitChoice`](@ref); the kernel
stores the full `(cell..., destination)` probability tensor populated
by the backward pass.
"""
struct Migration{Cmat<:AbstractMatrix, T<:Real, N, D,
                 L<:StateLayout, AV<:AbstractArray{T,N}} <: AbstractStage
    location_axis  :: Symbol
    location_dim   :: Int
    migration_cost :: Cmat
    ε              :: Param{T}
    closure_deps   :: NTuple{D, Symbol}
    input_layout   :: L
    output_layout  :: L
    V_start        :: AV
    Λ_end          :: AV
end

"""CLAUDE
Construct a [`Migration`](@ref) stage on `layout`. `migration_cost` is
the `(n_loc, n_loc)` cost matrix (row = origin, column = destination);
`ε` is a `Param{T}` or raw number for the Gumbel scale.
"""
function Migration(layout::StateLayout;
                   location_axis::Symbol = :location,
                   migration_cost::AbstractMatrix,
                   ε,
                   closure_deps::NTuple{D, Symbol} = (),
                   element_type::Union{Type, Nothing} = nothing,
                   V_start::Union{Nothing, AbstractArray} = nothing,
                   Λ_end::Union{Nothing, AbstractArray}   = nothing) where {D}
    location_dim = axis_position(layout, location_axis)
    n_loc = axissize(layout.axes[location_dim])
    size(migration_cost) == (n_loc, n_loc) ||
        error("Migration: cost matrix has shape $(size(migration_cost)), " *
              "expected ($n_loc, $n_loc) for axis :$location_axis of size $n_loc")
    ε_param = ε isa Param ? ε : Param(Float64(ε))
    T_default = let v = ε_param.val
        v isa Symbol ? Float64 : (typeof(v) <: Real ? typeof(v) : Float64)
    end
    T = @something element_type T_default
    dims = layout_size(layout)
    Vs   = @something V_start zeros(T, dims)
    Λe   = @something Λ_end   zeros(T, dims)
    @assert typeof(Vs) === typeof(Λe) "Migration: V_start and Λ_end must have the same concrete array type"
    return Migration{typeof(migration_cost), T, length(dims), D,
                     typeof(layout), typeof(Vs)}(
        location_axis, location_dim, migration_cost, ε_param, closure_deps,
        layout, layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:Migration}) = NamedTuple()

function allocate(stage::Migration{Cmat,T,N},
                  ::Type{T2} = T) where {Cmat,T,N,T2}
    dims  = layout_size(stage.input_layout)
    n_loc = axissize(stage.input_layout.axes[stage.location_dim])
    prob  = zeros(T2, dims..., n_loc)
    return ((choice_prob = prob,), nothing)
end

# Backward #
#----------#
# V_pre[..., i, ...] = ε log Σ_j exp((-C[i, j] + V_post[..., j, ...]) / ε)
# with `i` and `j` running over the location-axis positions and `...`
# over the non-location state.

function backward!(stage::Migration{Cmat,T,N},
                   V_end::AbstractArray{T,N},
                   env, kernel, scratch) where {Cmat,T,N}
    layout = stage.input_layout
    ldim   = stage.location_dim
    n_loc  = axissize(layout.axes[ldim])
    C      = stage.migration_cost
    ε      = resolve(stage.ε, env)
    V_start = stage.V_start
    prob    = kernel.choice_prob
    dims    = layout_size(layout)

    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        i_loc   = in_idxs[ldim]

        # Pass 1: max for numerical stability.
        max_u = typemin(T)
        for j in 1:n_loc
            out_idxs = Base.setindex(in_idxs, j, ldim)
            u = -C[i_loc, j] + V_end[CartesianIndex(out_idxs)]
            u > max_u && (max_u = u)
        end

        # Pass 2: unnormalised weights.
        denom = zero(T)
        for j in 1:n_loc
            out_idxs = Base.setindex(in_idxs, j, ldim)
            u = -C[i_loc, j] + V_end[CartesianIndex(out_idxs)]
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
    return V_start
end

# Forward #
#---------#
# Λ_post[j, s] = Σ_i P(j | i, s) · Λ_pre[i, s]

function forward!(stage::Migration{Cmat,T,N},
                  Λ_start::AbstractArray{T,N},
                  kernel, scratch,
                  moments = nothing) where {Cmat,T,N}
    layout = stage.input_layout
    ldim   = stage.location_dim
    n_loc  = axissize(layout.axes[ldim])
    Λ_end  = stage.Λ_end
    prob   = kernel.choice_prob
    dims   = layout_size(layout)

    fill!(Λ_end, zero(T))
    for ci in CartesianIndices(dims)
        in_idxs = Tuple(ci)
        mass    = Λ_start[ci]
        iszero(mass) && continue
        for j in 1:n_loc
            p = prob[in_idxs..., j]
            iszero(p) && continue
            out_idxs = Base.setindex(in_idxs, j, ldim)
            Λ_end[CartesianIndex(out_idxs)] += mass * p
        end
    end
    return Λ_end
end
