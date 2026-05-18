using Test
using HouseholdStages

# These tests guard the cell-positional / env-kwarg convention without
# the prior `PayoffFn` wrapper. Functions broadcast as scalars by
# default in modern Julia (`Base.Broadcast.broadcastable(::Function)`
# returns a `Ref`), so stage internals like `_fill_wealth_post!` can
# call `closure.(cells_arr; env = Ref(env))` and have the closure
# treated as a scalar across the broadcast.

@testset "closure call convention — cell positional, env kwarg" begin
    u = (cell, c; env) -> c^2 * env.σ + cell.bonus
    @test u((bonus = 10.0,), 1.5; env = (σ = 1.0,)) ≈ 12.25
    fp = (cell, a; env) -> cell.x + a * env.r
    @test fp((x = 10.0,), 2.0; env = (r = 0.5,)) ≈ 11.0
end

@testset "closure broadcasts as scalar over a cell array" begin
    # The library broadcasts closures as scalars (e.g.,
    # `wealth_post.(cells_arr; env = Ref(env))` in `_fill_wealth_post!`).
    # In that pattern the closure receives `env` as a `Ref` and unwraps it
    # with `env[]`; the broadcast-as-scalar behavior is what we test here.
    u = (cell, c; env) -> c^2 * env[].σ + cell.wealth
    cells = [(wealth = 1.0,), (wealth = 2.0,), (wealth = 3.0,)]
    out = u.(cells, 2.0; env = Ref((σ = 1.0,)))
    @test out == [5.0, 6.0, 7.0]
end

@testset "closure is type-stable at the call site" begin
    no_cell  = (cell, c; env) -> c * env.r
    yes_cell = (cell, c; env) -> c * env.r + cell.b
    cell = (b = 1.0,)
    env  = (r = 2.0,)
    @inferred no_cell(cell, 0.5; env = env)
    @inferred yes_cell(cell, 0.5; env = env)
end
