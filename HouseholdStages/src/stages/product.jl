"""
Configuration for a Cartesian product of `n_components` *uniform*
stages along a new product axis. The product axis is appended to the
component layout as a `discrete_finite(1:n_components)` axis with the
given name.

The K-operator is the block-diagonal direct sum `⊕_i K_i`; backward
and forward operate per-component on the corresponding slice of the
fused tensor.

v1 restricts to components with identical concrete Spec type and
identical `input_layout`. Heterogeneous-shape products are stubbed at
`src/stages/heterogeneous_product/`.
"""
struct ProductStageSpec{Specs<:Tuple, T<:Real,
                        LComp<:StateLayout, LProd<:StateLayout} <: AbstractStageSpec
    components       :: Specs
    axis             :: Symbol
    axis_size        :: Int
    component_layout :: LComp
    input_layout     :: LProd
    output_layout    :: LProd
    element_type     :: Type{T}
end

"""
Per-call buffer for a `ProductStage`. Holds two fused tensors of
shape `(component_dims..., n_components)` and a tuple of per-component
Buffers. Component buffers' `V_start` / `Λ_end` are views into the
fused tensors so backward/forward write directly with no copies.
Asymmetric-layout components (e.g., `ForgetfulSumStage`) keep V as
a view but fall back to fresh-allocated Λ_end — see
[`_accepts_view_Λ`](@ref).
"""
struct ProductStageBuffer{Buffers<:Tuple, T<:Real, Nfused} <: AbstractStageBuffer
    components :: Buffers
    V_fused    :: Array{T, Nfused}
    Λ_fused    :: Array{T, Nfused}
    cache      :: CacheState
end

"""
A Cartesian product of uniform stages along a new product axis.
Construct via `product(s1, s2, …; axis=:group)` or `s1 × s2` (sugar
for `product(s1, s2; axis=:group)`).
"""
struct ProductStage{Spec<:ProductStageSpec, Buffer<:ProductStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

# Build a Spec from a tuple of component Specs.
function ProductStageSpec(components::Tuple; axis::Symbol = :group,
                          element_type::Type{T} = _component_eltype(components)) where {T<:Real}
    isempty(components) && error("ProductStageSpec: at least one component required")
    first_type = typeof(components[1])
    if !all(s -> typeof(s) === first_type, components)
        error("ProductStageSpec: heterogeneous component types not supported in v1; " *
              "see src/stages/heterogeneous_product/README.md")
    end
    first_layout = components[1].input_layout
    if !all(s -> s.input_layout === first_layout || _layout_equal(s.input_layout, first_layout),
            components)
        error("ProductStageSpec: heterogeneous component layouts not supported in v1; " *
              "see src/stages/heterogeneous_product/README.md")
    end

    n        = length(components)
    new_axis = StateAxis(axis, discrete_finite(collect(1:n)))
    prod_lay = StateLayout(first_layout.axes..., new_axis)
    return ProductStageSpec{typeof(components), T,
                            typeof(first_layout), typeof(prod_lay)}(
        components, axis, n, first_layout, prod_lay, prod_lay, element_type,
    )
end

# Cheap structural comparison of layouts: same names and same axissizes.
function _layout_equal(a::StateLayout, b::StateLayout)
    axisnames(a) === axisnames(b) || return false
    return layout_size(a) == layout_size(b)
end

# Element type from the first component's element_type field (every Spec
# carries one). Falls back to Float64 if absent.
function _component_eltype(components::Tuple)
    spec1 = components[1]
    return hasfield(typeof(spec1), :element_type) ? spec1.element_type : Float64
end

"""
    product(stages::AbstractStage...; axis=:group) -> ProductStage

Build a `ProductStage` from uniform bundled stages. The Spec is
composed of the components' Specs; the Buffer is freshly allocated
with view-stitching where possible.
"""
function product(stages::AbstractStage...; axis::Symbol = :group)
    isempty(stages) && error("product: at least one component required")
    specs = map(s -> s.spec, stages)
    spec  = ProductStageSpec(specs; axis)
    return ProductStage(spec, allocate(spec))
end

"""
    product(specs::AbstractStageSpec...; axis=:group) -> ProductStageSpec

Spec-level product. Returns a fresh `ProductStageSpec`; no allocation.
"""
function product_spec(specs::AbstractStageSpec...; axis::Symbol = :group)
    return ProductStageSpec(specs; axis)
end

ProductStage(spec::ProductStageSpec) = ProductStage(spec, allocate(spec))
bundle(spec::ProductStageSpec)       = ProductStage(spec)

# Allocate #
#----------#

function allocate(spec::ProductStageSpec{Specs, Tspec, LComp, LProd},
                  ::Type{T} = spec.element_type) where {Specs, Tspec, LComp, LProd, T}
    n          = spec.axis_size
    dims_fused = layout_size(spec.input_layout)
    Nfused     = length(dims_fused)
    V_fused    = zeros(T, dims_fused)
    Λ_fused    = zeros(T, dims_fused)

    # Per-component buffers with view-stitching. Every concrete Spec's
    # `allocate(spec, T; V_start, Λ_end)` accepts pre-allocated buffers
    # (a uniform convention introduced by the 2026-05-19 refactor), so
    # view-stitching is universally supported on the V side. Λ requires
    # a shape check (`_accepts_view_Λ`) because ForgetfulSum's output
    # layout differs from its input layout.
    component_buffers = ntuple(n) do i
        comp_spec = spec.components[i]
        V_slice = selectdim(V_fused, Nfused, i)
        Λ_slice = selectdim(Λ_fused, Nfused, i)
        if _accepts_view_Λ(comp_spec, Λ_slice)
            allocate(comp_spec, T; V_start = V_slice, Λ_end = Λ_slice)
        else
            allocate(comp_spec, T; V_start = V_slice)
        end
    end

    return ProductStageBuffer{typeof(component_buffers), T, Nfused}(
        component_buffers, V_fused, Λ_fused, CacheState(),
    )
end

"""CLAUDE
Does the component Spec accept a `Λ_end` view of the given shape?
Default: yes (symmetric layout). `ForgetfulSumStageSpec` overrides to
return `false` (its Λ_end shape differs from the input-layout slice).
""" #CLAUDE Inline this
_accepts_view_Λ(spec::AbstractStageSpec, Λ_slice) =
    layout_size(spec.output_layout) == size(Λ_slice)

# Env slice — union over components #
#-----------------------------------#

static_env_deps(::Type{<:ProductStageSpec}) = NamedTuple()

function effective_env_slice(spec::ProductStageSpec)
    names = Symbol[]
    for s in spec.components
        for k in effective_env_slice(s)
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end

# Backward / forward — per-component on slices of the fused tensor #
#------------------------------------------------------------------#

function backward!(spec::ProductStageSpec{Specs, Tspec, LComp, LProd},
                   V_end, env, buffer::ProductStageBuffer) where {Specs, Tspec, LComp, LProd}
    n = spec.axis_size
    Nfused = ndims(V_end)
    # If first component's V_start is a view into V_fused, results land
    # there directly. Otherwise copy each component result back.
    views_into_fused = buffer.components[1].V_start isa SubArray
    for i in 1:n
        V_slice = selectdim(V_end, Nfused, i)
        V_start_comp = backward!(spec.components[i], V_slice, env, buffer.components[i])
        if !views_into_fused
            selectdim(buffer.V_fused, Nfused, i) .= V_start_comp
        end
    end
    _seat_cache!(buffer, V_end, env)
    return buffer.V_fused
end

function forward!(spec::ProductStageSpec{Specs, Tspec, LComp, LProd},
                  Λ_start, buffer::ProductStageBuffer) where {Specs, Tspec, LComp, LProd}
    n = spec.axis_size
    Nfused = ndims(Λ_start)
    views_into_fused = buffer.components[1].Λ_end isa SubArray
    for i in 1:n
        Λ_slice = selectdim(Λ_start, Nfused, i)
        Λ_end_comp = forward!(spec.components[i], Λ_slice, buffer.components[i])
        if !views_into_fused
            selectdim(buffer.Λ_fused, Nfused, i) .= Λ_end_comp
        end
    end
    return buffer.Λ_fused
end

# V_start / Λ_end accessors — the fused tensors #
#-----------------------------------------------#

V_start_buffer(stage::ProductStage) = stage.buffer.V_fused
Λ_end_buffer(stage::ProductStage)   = stage.buffer.Λ_fused

# Cache invalidation walks components #
#-------------------------------------#

function invalidate!(buffer::ProductStageBuffer)
    buffer.cache.kernel_valid = false
    for b in buffer.components
        invalidate!(b)
    end
    return buffer
end

# `×` operator — Spec-level primary, Stage-level sugar #
#------------------------------------------------------#

"""
    s1 × s2

Product of two stages along the default `:group` axis. Like `∘`,
defined on `AbstractStageSpec` (no allocation) and on `AbstractStage`
(sugar that bundles a fresh buffer).
"""
×(a::AbstractStageSpec, b::AbstractStageSpec) =
    ProductStageSpec((a, b); axis = :group)

×(a::AbstractStage, b::AbstractStage) = bundle(a.spec × b.spec)

# Legacy alias retained for one cycle (`×ₛ` predates the 2026-05-19 refactor).
const ×ₛ = ×

"""
    replicate_age(stage, N::Int; axis::Symbol = :age) -> ProductStage

Produce `N` uniform copies of `stage` joined along `axis` (default
`:age`). Equivalent to `product(stage, stage, …, stage; axis = axis)`.
Accepts a bundled `<X>Stage` or an `<X>StageSpec`.
"""
function replicate_age(stage::AbstractStage, N::Int; axis::Symbol = :age)
    return product(ntuple(_ -> stage, N)...; axis = axis)
end
function replicate_age(spec::AbstractStageSpec, N::Int; axis::Symbol = :age)
    return ProductStageSpec(ntuple(_ -> spec, N); axis = axis)
end
