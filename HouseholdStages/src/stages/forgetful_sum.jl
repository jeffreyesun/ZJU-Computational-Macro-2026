"""
    ForgetfulSumStage{T, Nin, Nout, LIn, LOut} <: AbstractStage

A layout-changing stage that drops one axis of the state space — pooling
households who are otherwise identical along the dropped axis. The
output layout is `input_layout` with `forget_axis` removed.

The K-operator is the sum-along-axis operator, fully determined by the
layout difference (no V/θ dependence; kernel = `nothing`). Backward
broadcasts `V_end` along the dropped axis; forward sums `Λ_start` along
it. Duality:

```
⟨V_in, Λ_in⟩  =  ⟨V_out, Λ_out⟩
```

(flow payoff `r = 0`).
"""
struct ForgetfulSumStage{T<:Real, Nin, Nout,
                    LIn<:StateLayout, LOut<:StateLayout,
                    AVin<:AbstractArray{T,Nin},
                    AVout<:AbstractArray{T,Nout}} <: AbstractStage
    forget_axis   :: Symbol
    forget_dim    :: Int
    input_layout  :: LIn
    output_layout :: LOut
    V_start       :: AVin
    Λ_end         :: AVout
end

"""
    ForgetfulSumStage(layout; forget_axis, element_type=Float64,
                          V_start=nothing, Λ_end=nothing) -> ForgetfulSumStage

Construct a stage that drops `forget_axis` from `layout`. Optional
`V_start` / `Λ_end` kwargs accept pre-allocated buffers — typically
views into a fused tensor when this stage lives inside a
[`ProductStage`](@ref).
"""
function ForgetfulSumStage(layout::StateLayout;
                      forget_axis::Symbol,
                      element_type::Type{T} = Float64,
                      V_start::Union{Nothing, AbstractArray} = nothing,
                      Λ_end::Union{Nothing, AbstractArray}  = nothing) where {T<:Real}
    forget_dim = axis_position(layout, forget_axis)
    output_layout = drop_axis(layout, forget_axis)
    dims_in  = layout_size(layout)
    dims_out = layout_size(output_layout)
    Vs = V_start === nothing ? zeros(T, dims_in)  : V_start
    Λe = Λ_end   === nothing ? zeros(T, dims_out) : Λ_end
    return ForgetfulSumStage{T, length(dims_in), length(dims_out),
                        typeof(layout), typeof(output_layout),
                        typeof(Vs), typeof(Λe)}(
        forget_axis, forget_dim, layout, output_layout, Vs, Λe,
    )
end

static_env_deps(::Type{<:ForgetfulSumStage}) = NamedTuple()

# K = sum-along-axis; no materialized kernel data, no scratch.
allocate(::ForgetfulSumStage, ::Type = Float64) = (; kernel = nothing, scratch = nothing)

# Backward: broadcast V_end along the dropped axis #
#--------------------------------------------------#

function backward!(stage::ForgetfulSumStage, V_end, env, buffers)
    (; output_layout, forget_dim, V_start) = stage
    dims_out = layout_size(output_layout)
    @assert size(V_end) == dims_out
    shape          = _insert_singleton(dims_out, forget_dim)
    V_end_reshaped = reshape(V_end, shape)
    V_start .= V_end_reshaped
    return V_start
end

# Forward: sum Λ_start along the dropped axis #
#---------------------------------------------#

function forward!(stage::ForgetfulSumStage, Λ_start, buffers, moments = nothing)
    (; input_layout, output_layout, forget_dim, Λ_end) = stage
    dims_in  = layout_size(input_layout)
    dims_out = layout_size(output_layout)
    @assert size(Λ_start) == dims_in
    shape          = _insert_singleton(dims_out, forget_dim)
    Λ_end_reshaped = reshape(Λ_end, shape)
    sum!(Λ_end_reshaped, Λ_start)
    return Λ_end
end

# Internals #
#-----------#

function _insert_singleton(dims::NTuple{N,Int}, pos::Int) where {N}
    @assert 1 <= pos <= N + 1
    return ntuple(i -> i < pos  ? dims[i]   :
                       i == pos ? 1         :
                                  dims[i-1], Val(N+1))
end
