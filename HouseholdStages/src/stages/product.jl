"""
Cartesian product of `n_components` *uniform* stages along a new product
axis. The product axis is appended to the component layout as a
`discrete_finite(1:n_components)` axis with the given name. The K-operator
is the block-diagonal direct sum `⊕_i K_i`; backward and forward operate
per-component on the corresponding slice of the fused tensor.

v1 restricts to components with identical Julia type and identical
`input_layout` (heterogeneous-shape products are stubbed at
`src/stages/heterogeneous_product/`). Components are rebuilt with
view-buffers into the fused tensors where possible, so backward/forward
write directly into the fused tensor without copies.
"""
struct ProductStage{Stages<:Tuple, T<:Real, Nfused,
                    LComp<:StateLayout, LProd<:StateLayout} <: AbstractStage
    components       :: Stages
    axis             :: Symbol
    axis_size        :: Int
    component_layout :: LComp
    input_layout     :: LProd
    output_layout    :: LProd
    V_start          :: Array{T, Nfused}
    Λ_end            :: Array{T, Nfused}
end

"""
Build a `ProductStage` over uniform components. Errors with a pointer to
the heterogeneous-product stub if components are non-uniform.
"""
function product(stages::AbstractStage...; axis::Symbol = :group,
                 use_views::Bool = true)
    isempty(stages) && error("product: at least one component required")
    first_type = typeof(stages[1])
    if !all(s -> typeof(s) === first_type, stages)
        error("product: heterogeneous component types not supported in v1; " *
              "see src/stages/heterogeneous_product/README.md")
    end
    first_layout = stages[1].input_layout
    if !all(s -> s.input_layout === first_layout || _layout_equal(s.input_layout, first_layout),
            stages)
        error("product: heterogeneous component layouts not supported in v1; " *
              "see src/stages/heterogeneous_product/README.md")
    end

    n        = length(stages)
    new_axis = StateAxis(axis, discrete_finite(collect(1:n)))
    prod_lay = StateLayout(first_layout.axes..., new_axis)
    dims_fused = layout_size(prod_lay)
    T = eltype(stages[1].V_start)
    V_fused = zeros(T, dims_fused)
    Λ_fused = zeros(T, dims_fused)

    # If every component is a MarkovStage stage and `use_views` is on,
    # rebuild components with view-buffers into the fused tensors so
    # backward!/forward! write directly into the fused tensor (zero copies).
    # Other stage types currently keep the copy-based path; they can be
    # ported by adding V_start/Λ_end kwargs to their constructors plus
    # the buffer-type parameter to their struct.
    components = if use_views && _supports_view_buffers(first_type)
        Nfused = length(dims_fused)
        ntuple(n) do i
            V_slice = selectdim(V_fused, Nfused, i)
            Λ_slice = selectdim(Λ_fused, Nfused, i)
            _rebuild_with_view_buffers(stages[i], V_slice, Λ_slice)
        end
    else
        Tuple(stages)
    end

    return ProductStage{typeof(components), T, length(dims_fused),
                        typeof(first_layout), typeof(prod_lay)}(
        components, axis, n, first_layout, prod_lay, prod_lay,
        V_fused, Λ_fused,
    )
end

# Cheap structural comparison of layouts: same names and same axissizes.
function _layout_equal(a::StateLayout, b::StateLayout)
    axisnames(a) === axisnames(b) || return false
    return layout_size(a) == layout_size(b)
end

# View-buffer support per stage type. Extend by adding more
# `_supports_view_buffers` / `_rebuild_with_view_buffers` methods.
_supports_view_buffers(::Type) = false
_supports_view_buffers(::Type{<:MarkovStage})   = true
_supports_view_buffers(::Type{<:IdentityStage}) = true
_supports_view_buffers(::Type{<:ForgetfulSumStage})  = true
_supports_view_buffers(::Type{<:ArgmaxStage})        = true
_supports_view_buffers(::Type{<:LogitChoiceStage})   = true

function _rebuild_with_view_buffers(s::MarkovStage, V_slice, Λ_slice)
    return MarkovStage(s.input_layout;
                       axis       = s.axis,
                       transition = s.transition,
                       V_start    = V_slice,
                       Λ_end      = Λ_slice)
end

function _rebuild_with_view_buffers(s::IdentityStage, V_slice, Λ_slice)
    return IdentityStage(s.input_layout;
                         element_type = eltype(V_slice),
                         V_start      = V_slice,
                         Λ_end        = Λ_slice)
end

# Note: `ForgetfulSumStage` is a layout-changing stage with `V_start` of
# input-layout shape and `Λ_end` of output-layout shape (different).
# Inside a product, the fused tensor has the *input* layout extended
# by the product axis — so V_slice (along product axis) has
# input-layout shape, matching V_start. But Λ_end has *output* shape,
# which is the input shape minus the forgotten axis; the product's
# Λ_fused has input-shape (because the product axis was appended to
# the INPUT layout). So we can't directly slice Λ_fused into the
# Λ_end shape. We fall back to a fresh allocation for Λ_end in this
# case — V_start uses a view, Λ_end is its own array. Half the
# copying is still saved.
function _rebuild_with_view_buffers(s::ForgetfulSumStage, V_slice, Λ_slice)
    # Λ_slice has the wrong shape (input-layout slice, not output);
    # ignore it and let ForgetfulSumStage allocate Λ_end fresh.
    return ForgetfulSumStage(s.input_layout;
                        forget_axis  = s.forget_axis,
                        element_type = eltype(V_slice),
                        V_start      = V_slice)
end

function _rebuild_with_view_buffers(s::ArgmaxStage, V_slice, Λ_slice)
    return ArgmaxStage(s.input_layout;
                  choice_axis    = s.choice_axis,
                  flow_payoff    = s.flow_payoff,
                  next_state_idx = s.next_state_idx,
                  element_type   = eltype(V_slice),
                  V_start        = V_slice,
                  Λ_end          = Λ_slice)
end

function _rebuild_with_view_buffers(s::LogitChoiceStage, V_slice, Λ_slice)
    return LogitChoiceStage(s.input_layout;
                       choice_axis    = s.choice_axis,
                       flow_payoff    = s.flow_payoff,
                       next_state_idx = s.next_state_idx,
                       ε              = s.ε,
                       element_type   = eltype(V_slice),
                       V_start        = V_slice,
                       Λ_end          = Λ_slice)
end

"""
    ×ₛ(a, b)

Infix product operator. `a ×ₛ b` is `product(a, b; axis = :group)`.
"""
×ₛ(a::AbstractStage, b::AbstractStage) = product(a, b; axis = :group)

"""
    replicate_age(stage, N::Int; axis::Symbol = :age) -> ProductStage

Produce `N` uniform copies of `stage` joined along `axis` (default
`:age`). Equivalent to `product(stage, stage, …, stage; axis = axis)`.
"""
function replicate_age(stage::AbstractStage, N::Int; axis::Symbol = :age)
    return product(ntuple(_ -> stage, N)...; axis = axis)
end

static_env_deps(::Type{<:ProductStage}) = NamedTuple()

# Effective env slice = union of per-component slices.
function effective_env_slice(ps::ProductStage)
    names = Symbol[]
    for s in ps.components
        for k in effective_env_slice(s)
            push!(names, k)
        end
    end
    return Tuple(unique(names))
end

# Allocate #
#----------#

# A product's workspace mirrors the chain shape: a `Tuple` of per-component
# `(; kernel, scratch)` bundles. `backward!` / `forward!` pass `buffers[i]`
# through to each component without unpacking.
function allocate(ps::ProductStage, ::Type{T2} = eltype(ps.V_start)) where {T2}
    return map(s -> allocate(s, T2), ps.components)
end

# Backward (per-component) #
#--------------------------#

function backward!(ps::ProductStage{Stages, T, Nfused},
                   V_end, env, buffers) where {Stages, T, Nfused}
    n = ps.axis_size
    # Detect once whether components use views into the fused tensor.
    # If yes, backward!(component, V_slice, ...) writes the result
    # directly into `ps.V_start[..., i]` — no copy-back needed.
    views_into_fused = ps.components[1].V_start isa SubArray
    for i in 1:n
        V_slice = selectdim(V_end, Nfused, i)
        V_start_comp = backward!(ps.components[i], V_slice, env, buffers[i])
        if !views_into_fused
            selectdim(ps.V_start, Nfused, i) .= V_start_comp
        end
    end
    return ps.V_start
end

# Forward (per-component) #
#-------------------------#

function forward!(ps::ProductStage{Stages, T, Nfused},
                  Λ_start, buffers, moments = nothing) where {Stages, T, Nfused}
    n = ps.axis_size
    views_into_fused = ps.components[1].Λ_end isa SubArray
    for i in 1:n
        Λ_slice = selectdim(Λ_start, Nfused, i)
        Λ_end_comp = forward!(ps.components[i], Λ_slice, buffers[i], moments)
        if !views_into_fused
            selectdim(ps.Λ_end, Nfused, i) .= Λ_end_comp
        end
    end
    return ps.Λ_end
end
