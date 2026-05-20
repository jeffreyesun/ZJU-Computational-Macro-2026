"""
Configuration for a [`BorrowingConstraintStage`](@ref): a
state-feasibility stage that marks a designated set of cells as
infeasible by setting `V_start[s] = -Inf` on the backward pass;
identity on Λ on the forward pass. Pure data — no per-call buffers.

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
    Spec and re-used every backward call. Cheap and GPU-portable.
  * **State / env-dependent closure.** Pass `infeasible::Function` of
    signature `(cell; env) -> Bool`. The mask is materialised on every
    backward pass into the buffer's kernel from `cell_array(layout)`
    and the runtime `env`. `env` is passed plainly to the closure
    (`env.w_min`, etc.) — no `Ref` wrapping.

Forward pass is identity on Λ; the forbidden-state mass is whatever
upstream put there (typically zero if the chain is consistent).
"""
struct BorrowingConstraintStageSpec{Inf_t, T<:Real, L<:StateLayout} <: AbstractStageSpec
    infeasible    :: Inf_t
    input_layout  :: L
    output_layout :: L
    element_type  :: Type{T}
end

"""
    BorrowingConstraintStageSpec(layout; infeasible, element_type=Float64)

Build the Spec for a [`BorrowingConstraintStage`](@ref). `infeasible`
is either an `AbstractArray{Bool}` of layout shape (`true` for
infeasible) or a `(cell; env) -> Bool` closure.
"""
function BorrowingConstraintStageSpec(layout::StateLayout;
                                      infeasible,
                                      element_type::Type{T} = Float64) where {T<:Real}
    inf_field = _wrap_infeasible(infeasible, layout_size(layout))
    return BorrowingConstraintStageSpec{typeof(inf_field), T, typeof(layout)}(
        inf_field, layout, layout, element_type,
    )
end

"""
Per-call buffer for a borrowing-constraint stage. For the precomputed
`AbstractArray{Bool}` form the kernel is `nothing` (the mask lives on
the Spec). For the closure form the kernel is `(; mask)` — a per-cell
`Bool` array re-materialised every `backward!`. Scratch is always
`nothing`.
"""
struct BorrowingConstraintStageBuffer{T<:Real, N, AV<:AbstractArray{T,N},
                                      Kernel} <: AbstractStageBuffer
    kernel  :: Kernel
    scratch :: Nothing
    V_start :: AV
    Λ_end   :: AV
    cache   :: CacheState
end

"""
A state-feasibility stage. Construct via `BorrowingConstraintStage(layout;
infeasible)`. `infeasible` is either an `AbstractArray{Bool}` of
layout shape or a `(cell; env) -> Bool` closure (see the Spec docstring).

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
struct BorrowingConstraintStage{Spec<:BorrowingConstraintStageSpec,
                                Buffer<:BorrowingConstraintStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

function BorrowingConstraintStage(layout::StateLayout;
                                  infeasible,
                                  element_type::Type{T} = Float64,
                                  V_start::Union{Nothing, AbstractArray} = nothing,
                                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {T<:Real}
    spec = BorrowingConstraintStageSpec(layout; infeasible, element_type)
    return BorrowingConstraintStage(spec, allocate(spec, T; V_start, Λ_end))
end

BorrowingConstraintStage(spec::BorrowingConstraintStageSpec) =
    BorrowingConstraintStage(spec, allocate(spec))
bundle(spec::BorrowingConstraintStageSpec) = BorrowingConstraintStage(spec)

static_env_deps(::Type{<:BorrowingConstraintStageSpec}) = NamedTuple()

"""CLAUDE
Normalise the `infeasible` argument into a stored field. Boolean arrays
pass through after a shape check; closures pass through unchanged
(functions broadcast as scalars by default).
""" #CLAUDE I thnk you can just inline this with an `infeasible isa AbstractArray && @assert size(infeasible) == layout_size(layout)` check.
function _wrap_infeasible(arr::AbstractArray{Bool}, dims::Tuple)
    size(arr) == dims ||
        error("BorrowingConstraintStage: infeasible mask has shape $(size(arr)), expected layout shape $dims")
    return arr
end
_wrap_infeasible(f, ::Tuple) = f

# Allocate #
#----------#
# Closure form needs a per-cell `Bool` buffer to materialise on every
# backward. Array form needs nothing: the mask lives on the Spec.

function allocate(spec::BorrowingConstraintStageSpec{Inf_t},
                  ::Type{T} = spec.element_type;
                  V_start::Union{Nothing, AbstractArray} = nothing,
                  Λ_end::Union{Nothing, AbstractArray}   = nothing) where {Inf_t, T}
    (; Vs, Λe) = _alloc_VΛ(spec.input_layout, T, V_start, Λ_end)
    if Inf_t <: AbstractArray
        kernel = nothing
    else
        dims   = layout_size(spec.input_layout)
        kernel = (; mask = Array{Bool}(undef, dims))
    end
    return BorrowingConstraintStageBuffer{T, ndims(Vs), typeof(Vs), typeof(kernel)}(
        kernel, nothing, Vs, Λe, CacheState(),
    )
end

# Backward #
#----------#
# Array form: read mask straight off the Spec; kernel is `nothing`.
function backward!(spec::BorrowingConstraintStageSpec{Inf_t, T},
                   V_end, env,
                   buffer::BorrowingConstraintStageBuffer) where {Inf_t <: AbstractArray, T}
    V_start = buffer.V_start
    mask    = spec.infeasible
    @. V_start = ifelse(mask, T(-Inf), V_end)
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Closure form: materialise mask into kernel.mask, then apply. The
# AbstractArray method above wins when `Inf_t` is a concrete array; this
# fallback catches everything else. Broadcasting captures `env` as a
# kwarg (not broadcast), so the closure still sees env unwrapped.
function backward!(spec::BorrowingConstraintStageSpec{Inf_t, T},
                   V_end, env,
                   buffer::BorrowingConstraintStageBuffer) where {Inf_t, T}
    (; kernel, V_start) = buffer
    mask      = kernel.mask
    cells_arr = cell_array(spec.input_layout)
    mask .= spec.infeasible.(cells_arr; env)
    @. V_start = ifelse(mask, T(-Inf), V_end)
    _seat_cache!(buffer, V_end, env)
    return V_start
end

# Forward #
#---------#
# Identity on Λ — the constraint lives entirely on the V side.

function forward!(spec::BorrowingConstraintStageSpec, Λ_start,
                  buffer::BorrowingConstraintStageBuffer)
    copyto!(buffer.Λ_end, Λ_start)
    return buffer.Λ_end
end
