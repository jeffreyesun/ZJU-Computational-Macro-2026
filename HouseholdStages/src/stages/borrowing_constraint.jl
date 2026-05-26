"""
State-feasibility stage. Backward sets `V_start[s] = -Inf` on
infeasible cells, identity on `V_end` otherwise. Forward is identity
on Λ. `infeasible` is either an `AbstractArray{Bool}` of layout
shape or a `(cell; env) -> Bool` closure.
"""
struct BorrowingConstraintStageSpec{Inf_t} <: AbstractStageSpec
    infeasible :: Inf_t
end

BorrowingConstraintStageSpec(; infeasible) =
    BorrowingConstraintStageSpec{typeof(infeasible)}(infeasible)

"Kernel: a Bool mask of layout shape. Aliases `spec.infeasible` when it's an array; a fresh buffer that `backward!` refreshes when it's a closure."
struct BorrowingConstraintKernel{M<:AbstractArray{Bool}}
    mask :: M
end

function allocate_kernel(spec::BorrowingConstraintStageSpec, ::Type, layout::StateLayout)
    dims = layout_size(layout)
    mask = if spec.infeasible isa AbstractArray{Bool}
        @assert size(spec.infeasible) == dims
        spec.infeasible
    else
        Array{Bool}(undef, dims)
    end
    return BorrowingConstraintKernel(mask)
end

# Backward / forward #
#--------------------#

function backward!(buffer, spec::BorrowingConstraintStageSpec, V_end, env)
    T    = eltype(buffer.V_start)
    mask = buffer.kernel.mask
    spec.infeasible isa AbstractArray{Bool} ||
        (mask .= spec.infeasible.(cell_array(buffer.input_layout); env))
    @. buffer.V_start = ifelse(mask, T(-Inf), V_end)
    _seat_cache!(buffer, V_end, env)
    return buffer.V_start
end

function forward!(buffer, spec::BorrowingConstraintStageSpec, Λ_start)
    copyto!(buffer.Λ_end, Λ_start)
    return buffer.Λ_end
end

# Wrapper #
#---------#

@definestage BorrowingConstraintStage BorrowingConstraintStageSpec kernel=BorrowingConstraintKernel
