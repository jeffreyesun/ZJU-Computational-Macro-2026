using Test
using HouseholdStages

# Guards the cell-positional / env-kwarg closure convention used
# throughout the library. Stage internals (e.g., `_fill_wealth_post!`)
# call the closure once per cell in a plain loop, passing `env` directly
# (no `Ref` wrapper). Functions broadcast as scalars by default in
# modern Julia, so a closure-broadcast call site like `f.(cells_arr;
# env = env)` would also work — but we only rely on direct invocation.

@testset "closure call convention — cell positional, env kwarg" begin
    u = (cell, c; env) -> c^2 * env.σ + cell.bonus
    @test u((bonus = 10.0,), 1.5; env = (σ = 1.0,)) ≈ 12.25
    fp = (cell, a; env) -> cell.x + a * env.r
    @test fp((x = 10.0,), 2.0; env = (r = 0.5,)) ≈ 11.0
end

@testset "closure can be looped over a cell array with env passed directly" begin
    u = (cell, c; env) -> c^2 * env.σ + cell.wealth
    cells = [(wealth = 1.0,), (wealth = 2.0,), (wealth = 3.0,)]
    env = (σ = 1.0,)
    out = [u(cell, 2.0; env = env) for cell in cells]
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
