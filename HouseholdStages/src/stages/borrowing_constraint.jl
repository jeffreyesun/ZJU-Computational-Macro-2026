"""
    BorrowingConstraint{Inf_t, T, N, L<:StateLayout, D,
                        AV<:AbstractArray{T,N}} <: AbstractStage

State-feasibility stage. Marks a designated set of cells as infeasible
by setting `V_start[s] = -Inf` on the backward pass; identity on Λ on
the forward pass.

Operationally,

```
V_start[s] = -Inf  if  infeasible(s; env)
V_start[s] = V_end[s]  otherwise
```

This is the K-operator-level encoding of a *hard* constraint
(borrowing constraint, lower-wealth bound, loan-to-value rule, …): an
upstream optimization that sees `V = -Inf` will never target the
infeasible cell, so the constraint is enforced by the value-function
side of the duality without changing K.

Two construction patterns:

  * **Precomputed mask.** Pass `infeasible::AbstractArray{Bool}` of
    layout shape (`true` ≡ infeasible). The array is stored on the
    struct and re-used every backward call. Cheap and GPU-portable.
  * **State / env-dependent closure.** Pass `infeasible::Function` of
    signature `(cell; env) -> Bool`. The mask is materialised on every
    backward pass from `cell_array(layout)` and the runtime `env`.
    `closure_deps` should list the env fields read by the closure.

    `env` is passed as a `Ref` (matching the [`WealthChange`](@ref)
    convention), so the closure unwraps with `env[]`:
    `(cell; env) -> cell.wealth < env[].w_min`.

Forward pass is identity on Λ; the forbidden-state mass is whatever
upstream put there (typically zero if the chain is consistent).

# Examples

```julia
# Precomputed lower-wealth bound on a 2D layout:
mask = falses(layout_size(layout))
mask[1:3, :] .= true                                    # first three wealth cells infeasible
stage = BorrowingConstraint(layout; infeasible = mask)

# State-dependent loan-to-value: infeasible if `wealth < -ltv * q * h`.
stage = BorrowingConstraint(layout;
    infeasible = (cell; env) -> begin
        e = env[]
        cell.wealth < -e.ltv * e.q * cell.h
    end,
    closure_deps = (:ltv, :q),
)
```
"""
struct BorrowingConstraint{Inf_t, T<:Real, N, L<:StateLayout, D,
                           AV<:AbstractArray{T, N}} <: AbstractStage
    infeasible    :: Inf_t
    closure_deps  :: NTuple{D, Symbol}
    input_layout  :: L
    output_layout :: L
    V_start       :: AV
    Λ_end         :: AV
end

"""CLAUDE
Construct a [`BorrowingConstraint`](@ref) on `layout`. `infeasible` is
either an `AbstractArray{Bool}` of layout shape (`true` for infeasible)
or a `(cell; env) -> Bool` closure. `closure_deps` should list the env
fields read by the closure form; it is ignored for the array form.
"""
function BorrowingConstraint(layout::StateLayout;
                             infeasible,
                             closure_deps::NTuple{D, Symbol} = (),
                             element_type::Type{T} = Float64,
                             V_start::Union{Nothing, AbstractArray} = nothing,
                             Λ_end::Union{Nothing, AbstractArray}   = nothing) where {D, T<:Real}
    dims = layout_size(layout)
    Vs   = @something V_start zeros(T, dims)
    Λe   = @something Λ_end   zeros(T, dims)
    @assert typeof(Vs) === typeof(Λe) "BorrowingConstraint: V_start and Λ_end must have the same concrete array type"
    inf_field = _wrap_infeasible(infeasible, dims)
    N = length(dims)
    return BorrowingConstraint{typeof(inf_field), T, N, typeof(layout), D, typeof(Vs)}(
        inf_field, closure_deps, layout, layout, Vs, Λe,
    )
end

"""CLAUDE
Normalise the `infeasible` argument into a stored field. Boolean arrays
pass through after a shape check; closures pass through unchanged
(functions broadcast as scalars by default).
"""
function _wrap_infeasible(arr::AbstractArray{Bool}, dims::Tuple)
    size(arr) == dims ||
        error("BorrowingConstraint: infeasible mask has shape $(size(arr)), expected layout shape $dims")
    return arr
end
_wrap_infeasible(f, ::Tuple) = f

static_env_deps(::Type{<:BorrowingConstraint}) = NamedTuple()

# Allocation #
#------------#
# Closure form needs a per-cell `Bool` buffer to materialise on every
# backward. Array form needs nothing: the mask lives on the struct.

function allocate(stage::BorrowingConstraint{Inf_t, T, N},
                  ::Type{T2} = T) where {Inf_t, T, N, T2}
    if Inf_t <: AbstractArray
        return (nothing, nothing)
    else
        dims = layout_size(stage.input_layout)
        return ((mask = Array{Bool}(undef, dims),), nothing)
    end
end

# Backward #
#----------#
# Array form: read mask straight off the struct.
function backward!(stage::BorrowingConstraint{Inf_t, T, N},
                   V_end::AbstractArray{T, N},
                   env, kernel, scratch) where {Inf_t <: AbstractArray, T, N}
    V_start = stage.V_start
    mask    = stage.infeasible
    @. V_start = ifelse(mask, T(-Inf), V_end)
    return V_start
end

# Closure form: materialise mask into kernel.mask, then apply. The
# closure dispatch is implicit — the AbstractArray method above wins
# when `Inf_t` is concrete-array; this fallback catches everything else.
function backward!(stage::BorrowingConstraint{Inf_t, T, N},
                   V_end::AbstractArray{T, N},
                   env, kernel, scratch) where {Inf_t, T, N}
    V_start = stage.V_start
    mask    = kernel.mask
    cells_arr = cell_array(stage.input_layout)
    mask .= stage.infeasible.(cells_arr; env = Ref(env))
    @. V_start = ifelse(mask, T(-Inf), V_end)
    return V_start
end

# Forward #
#---------#
# Identity on Λ — the constraint lives entirely on the V side.

function forward!(stage::BorrowingConstraint{Inf_t, T, N},
                  Λ_start::AbstractArray{T, N},
                  kernel, scratch,
                  moments = nothing) where {Inf_t, T, N}
    copyto!(stage.Λ_end, Λ_start)
    return stage.Λ_end
end
