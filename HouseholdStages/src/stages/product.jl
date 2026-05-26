"""
Cartesian product of `n` *uniform* stages along a new product axis.
The K-operator is the block-diagonal direct sum `⊕_i K_i`. The
product axis is appended to the component layout as a
`discrete_finite(1:n)` axis with the user-chosen name.

v1 restricts to components with identical concrete Spec type and
identical input layout (when allocated against the product layout).
"""
struct ProductStageSpec{Specs<:Tuple} <: AbstractStageSpec
    components :: Specs
    axis       :: Symbol
end

function ProductStageSpec(components::Tuple; axis::Symbol=:group)
    @assert !isempty(components)
    first_type = typeof(components[1])
    @assert all(s -> typeof(s) === first_type, components)
    return ProductStageSpec{typeof(components)}(components, axis)
end

"Buffer for a product. Holds the fused `V` and `Λ` tensors plus per-component sub-buffers (views into the fused tensors when possible)."
struct ProductStageBuffer{Buffers<:Tuple, T, Nfused, LIn, LOut} <: AbstractStageBuffer
    components    :: Buffers
    V_fused       :: Array{T, Nfused}
    Λ_fused       :: Array{T, Nfused}
    input_layout  :: LIn
    output_layout :: LOut
    cache         :: CacheState
end

"Product stage: bundled stages of identical type joined along a new axis."
struct ProductStage{Spec<:ProductStageSpec, Buffer<:ProductStageBuffer} <: AbstractStage
    spec   :: Spec
    buffer :: Buffer
end

# Construct a ProductStage from bundled sub-stages.
function product(stages::AbstractStage...; axis::Symbol=:group)
    @assert !isempty(stages)
    specs  = map(s -> s.spec, stages)
    spec   = ProductStageSpec(specs; axis)
    # All components share an input layout: pick the first.
    comp_layout = stages[1].buffer.input_layout
    return ProductStage(spec, comp_layout)
end

# Spec-level product (no allocation).
product_spec(specs::AbstractStageSpec...; axis::Symbol=:group) =
    ProductStageSpec(specs; axis)

ProductStage(spec::ProductStageSpec, comp_layout::StateLayout, ::Type{T}=Float64) where {T} =
    ProductStage(spec, allocate(spec, _product_layout(spec, comp_layout), T))

"Wrap a component layout into the product layout by appending the product axis."
function _product_layout(spec::ProductStageSpec, comp_layout::StateLayout)
    n = length(spec.components)
    return StateLayout(comp_layout.axes..., StateAxis(spec.axis, discrete_finite(collect(1:n))))
end

bundle(spec::ProductStageSpec, layout::StateLayout) = ProductStage(spec, layout)
bundle(spec::ProductStageSpec, layout::StateLayout, ::Type{T}) where {T} =
    ProductStage(spec, layout, T)

# Allocate — view-stitching where possible #
#-----------------------------------------#

function allocate(spec::ProductStageSpec, layout::StateLayout,
                  ::Type{T}=Float64) where {T}
    n           = length(spec.components)
    comp_layout = _strip_product_axis(layout, spec.axis)
    dims_fused  = layout_size(layout)
    Nfused      = length(dims_fused)
    V_fused     = zeros(T, dims_fused)
    Λ_fused     = zeros(T, dims_fused)

    component_buffers = ntuple(n) do i
        comp_spec = spec.components[i]
        V_slice = selectdim(V_fused, Nfused, i)
        Λ_slice = selectdim(Λ_fused, Nfused, i)
        if _accepts_view_Λ(comp_spec, comp_layout, Λ_slice)
            allocate(comp_spec, comp_layout, T; V_start=V_slice, Λ_end=Λ_slice)
        else
            allocate(comp_spec, comp_layout, T; V_start=V_slice)
        end
    end
    return ProductStageBuffer(component_buffers, V_fused, Λ_fused,
                              layout, layout, CacheState())
end

"""
Strip the product axis from a fused layout, returning the per-component
layout. Asserts the product axis is the trailing one.
"""
function _strip_product_axis(layout::StateLayout, axis::Symbol)
    @assert axisname(last(layout.axes)) === axis
    return StateLayout(layout.axes[1:end-1]...)
end

"""
Does the component Spec produce an output layout of the same shape
as the V/Λ slice? Symmetric stages do; `ForgetfulSumStage` doesn't
(its output drops an axis) and falls back to a fresh-allocated `Λ_end`.
"""
function _accepts_view_Λ(spec::AbstractStageSpec, comp_layout::StateLayout, Λ_slice)
    return layout_size(output_layout(spec, comp_layout)) == size(Λ_slice)
end

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

function backward!(buffer::ProductStageBuffer, spec::ProductStageSpec, V_end, env)
    n      = length(spec.components)
    Nfused = ndims(V_end)
    # If first component's V_start is a view into V_fused, results land
    # there directly; otherwise copy back.
    views_into_fused = buffer.components[1].V_start isa SubArray
    for i in 1:n
        V_slice = selectdim(V_end, Nfused, i)
        V_start_comp = backward!(buffer.components[i], spec.components[i], V_slice, env)
        views_into_fused ||
            (selectdim(buffer.V_fused, Nfused, i) .= V_start_comp)
    end
    _seat_cache!(buffer, V_end, env)
    return buffer.V_fused
end

function forward!(buffer::ProductStageBuffer, spec::ProductStageSpec, Λ_start)
    n      = length(spec.components)
    Nfused = ndims(Λ_start)
    views_into_fused = buffer.components[1].Λ_end isa SubArray
    for i in 1:n
        Λ_slice = selectdim(Λ_start, Nfused, i)
        Λ_end_comp = forward!(buffer.components[i], spec.components[i], Λ_slice)
        views_into_fused ||
            (selectdim(buffer.Λ_fused, Nfused, i) .= Λ_end_comp)
    end
    return buffer.Λ_fused
end

# Endpoint accessors — fused tensors #
#------------------------------------#

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

Product of two stages along the default `:group` axis. Spec form
returns a `ProductStageSpec`; Stage form bundles a fresh buffer.
"""
×(a::AbstractStageSpec, b::AbstractStageSpec) =
    ProductStageSpec((a, b); axis=:group)

×(a::AbstractStage, b::AbstractStage) = product(a, b; axis=:group)

const ×ₛ = ×

"""
    replicate_age(stage, N; axis=:age) -> ProductStage

`N` uniform copies of `stage` joined along `axis`. Accepts a bundled
stage or a Spec.
"""
replicate_age(stage::AbstractStage, N::Int; axis::Symbol=:age) =
    product(ntuple(_ -> stage, N)...; axis=axis)
replicate_age(spec::AbstractStageSpec, N::Int; axis::Symbol=:age) =
    ProductStageSpec(ntuple(_ -> spec, N); axis=axis)
