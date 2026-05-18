using Test
using HouseholdStages

@testset "layout — kinds and constructors" begin
    g = continuous_grid(0.0, 10.0; size = 5)
    @test g isa ContinuousGrid
    @test length(g.grid) == 5
    @test g.grid[1] ≈ 0.0
    @test g.grid[end] ≈ 10.0

    g2 = continuous_grid([0.0, 0.5, 1.0])
    @test g2 isa ContinuousGrid
    @test g2.grid == [0.0, 0.5, 1.0]

    d_int = discrete_finite([1, 2, 3])
    @test d_int isa DiscreteFinite{Int}
    @test d_int.levels == [1, 2, 3]

    d_float = discrete_finite([0.0, 1.0, 2.0])
    @test d_float isa DiscreteFinite{Float64}

    d_sym = categorical([:NYC, :LA, :Chicago])
    @test d_sym isa DiscreteFinite{Symbol}
    @test d_sym.levels == [:NYC, :LA, :Chicago]
end

@testset "layout — StateAxis" begin
    ax = StateAxis(:wealth, continuous_grid(0.0, 1.0; size = 4))
    @test ax isa StateAxis
    @test axisname(ax) === :wealth
    @test axissize(ax) == 4
    @test length(axisvalues(ax)) == 4

    ax2 = StateAxis(:loc, categorical([:NYC, :LA]))
    @test axisname(ax2) === :loc
    @test axissize(ax2) == 2
    @test axisvalues(ax2) == [:NYC, :LA]
end

@testset "layout — StateLayout construction and axis_position" begin
    layout = StateLayout(
        StateAxis(:wealth,  continuous_grid(0.0, 100.0; size = 8)),
        StateAxis(:income,  discrete_finite([0.5, 1.0, 1.5])),
        StateAxis(:loc,     categorical([:NYC, :LA])),
    )
    @test length(layout) == 3
    @test axisnames(layout) === (:wealth, :income, :loc)
    @test layout_size(layout) == (8, 3, 2)

    @test axis_position(layout, :wealth) == 1
    @test axis_position(layout, :income) == 2
    @test axis_position(layout, :loc) == 3
    @test_throws ErrorException axis_position(layout, :nope)
end

@testset "layout — duplicate axis names error" begin
    @test_throws ErrorException StateLayout(
        StateAxis(:x, continuous_grid(0.0, 1.0; size = 2)),
        StateAxis(:x, discrete_finite([1, 2])),
    )
end

@testset "layout — cells iteration" begin
    layout = StateLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:loc,    categorical([:NYC, :LA])),
    )
    pairs = collect(cells(layout))
    @test length(pairs) == 6

    expected = [
        ((wealth=1, loc=1), (wealth=0.0, loc=:NYC)),
        ((wealth=2, loc=1), (wealth=1.0, loc=:NYC)),
        ((wealth=3, loc=1), (wealth=2.0, loc=:NYC)),
        ((wealth=1, loc=2), (wealth=0.0, loc=:LA)),
        ((wealth=2, loc=2), (wealth=1.0, loc=:LA)),
        ((wealth=3, loc=2), (wealth=2.0, loc=:LA)),
    ]
    for ((iexp, cexp), (igot, cgot)) in zip(expected, pairs)
        @test igot === iexp
        @test cgot === cexp
    end
end

@testset "layout — cells iteration eltype is concrete NamedTuple per leaf type" begin
    # All-Float64 layout: cell values should be NamedTuple of Floats.
    layout = StateLayout(
        StateAxis(:a, continuous_grid([0.0, 1.0])),
        StateAxis(:b, discrete_finite([10.0, 20.0])),
    )
    for (idx, cell) in cells(layout)
        @test idx isa NamedTuple{(:a, :b), <:NTuple{2, Int}}
        @test cell isa NamedTuple{(:a, :b), <:NTuple{2, Float64}}
    end
end

@testset "layout — cells iteration is type-stable" begin
    layout = StateLayout(
        StateAxis(:x, continuous_grid([0.0, 0.5, 1.0])),
        StateAxis(:y, categorical([:A, :B])),
    )
    it = cells(layout)
    # Iterate once and ensure inference yields a concrete return type.
    f(it) = begin
        s = 0.0
        for (idx, cell) in it
            s += cell.x
        end
        s
    end
    @inferred f(it)
    @test f(it) ≈ 3.0  # (0.0 + 0.5 + 1.0) * 2
end
