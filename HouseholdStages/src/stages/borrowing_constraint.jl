"""
    BorrowingConstraintStage{Inf_t, T, N, L<:StateLayout,
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
    `env` is passed plainly to the closure (`env.w_min`, etc.) — no
    `Ref` wrapping.

Forward pass is identity on Λ; the forbidden-state mass is whatever
upstream put there (typically zero if the chain is consistent).

# Examples

```julia
# Precomputed lower-wealth bound on a 2D layout:
mask = falses(layout_size(layout))
mask[1:3, :] .= true                                    # first three wealth cells infeasible
stage = BorrowingConstraintStage(layout; infeasible = mask)

# State-dependent loan-to-value: infeasible if `wealth < -ltv * q * h`.
stage = BorrowingConstraintStage(layout;
    infeasible = (cell; env) -> cell.wealth < -env.ltv * env.q * cell.h,
)
```
"""
struct BorrowingConstraintStage{Inf_t, T<:Real, N, L<:StateLayout,
                           AV<:AbstractArray{T, N}} <: AbstractStage
    infeasible    :: Inf_t
    input_layout  :: L
    output_layout :: L
    V_start       :: AV
    Λ_end         :: AV
end

"""CLAUDE
Construct a [`BorrowingConstraintStage`](@ref) on `layout`. `infeasible`
is either an `AbstractArray{Bool}` of layout shape (`true` for
infeasible) or a `(cell; env) -> Bool` closure.
"""
function BorrowingConstraintStage(layout::StateLayout;
                             infeasible,
                             element_type::Type{T} = Float64,
                             V_start::Union{Nothing, AbstractArray} = nothing,
                             Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T<:Real}
    (; Vs, Λe) = _alloc_VΛ(layout, T, V_start, Λ_end)
    inf_field = _wrap_infeasible(infeasible, size(Vs))
    return BorrowingConstraintStage{typeof(inf_field), T, ndims(Vs), typeof(layout), typeof(Vs)}(
        inf_field, layout, layout, Vs, Λe,
    )
end

"""CLAUDE
Normalise the `infeasible` argument into a stored field. Boolean arrays
pass through after a shape check; closures pass through unchanged
(functions broadcast as scalars by default).
"""
function _wrap_infeasible(arr::AbstractArray{Bool}, dims::Tuple)
    size(arr) == dims ||
        error("BorrowingConstraintStage: infeasible mask has shape $(size(arr)), expected layout shape $dims")
    return arr
end
_wrap_infeasible(f, ::Tuple) = f

static_env_deps(::Type{<:BorrowingConstraintStage}) = NamedTuple()

# Allocation #
#------------#
# Closure form needs a per-cell `Bool` buffer to materialise on every
# backward. Array form needs nothing: the mask lives on the struct.

function allocate(stage::BorrowingConstraintStage{Inf_t}, ::Type = Float64) where {Inf_t}
    if Inf_t <: AbstractArray
        return (; kernel = nothing, scratch = nothing)
    else
        dims = layout_size(stage.input_layout)
        return (; kernel = (; mask = Array{Bool}(undef, dims)), scratch = nothing)
    end
end

# Backward #
#----------#
# Array form: read mask straight off the struct.
function backward!(stage::BorrowingConstraintStage{Inf_t, T},
                   V_end, env, buffers) where {Inf_t <: AbstractArray, T}
    (; V_start, infeasible) = stage
    @. V_start = ifelse(infeasible, T(-Inf), V_end)
    return V_start
end

# Closure form: materialise mask into kernel.mask, then apply. The
# closure dispatch is implicit — the AbstractArray method above wins
# when `Inf_t` is concrete-array; this fallback catches everything else.
# Broadcasting captures `env` as a kwarg (not broadcast), so the closure
# still sees env unwrapped.
function backward!(stage::BorrowingConstraintStage{Inf_t, T},
                   V_end, env, buffers) where {Inf_t, T}
    (; kernel, scratch) = buffers
    (; V_start, input_layout) = stage
    mask      = kernel.mask
    cells_arr = cell_array(input_layout)
    mask .= stage.infeasible.(cells_arr; env)
    @. V_start = ifelse(mask, T(-Inf), V_end)
    return V_start
end

# Forward #
#---------#
# Identity on Λ — the constraint lives entirely on the V side.

function forward!(stage::BorrowingConstraintStage, Λ_start, buffers, moments = nothing)
    copyto!(stage.Λ_end, Λ_start)
    return stage.Λ_end
end
