using Test
using HouseholdStages

@testset "Param — calibrated literal" begin
    p = Param(0.25)
    @test p isa Param{Float64}
    @test p.val == 0.25
    @test !is_swept(p)
    @test swept_key(p) === nothing
    @test resolve(p, (a = 1.0,)) ≈ 0.25
end

@testset "Param — swept symbol" begin
    p = Param(:foo)
    @test p isa Param{Float64}
    @test p.val === :foo
    @test is_swept(p)
    @test swept_key(p) === :foo
    @test resolve(p, (foo = 0.5,)) ≈ 0.5
end

@testset "Param — explicit leaf type override" begin
    p = Param{Int}(:n_rooms)
    @test p isa Param{Int}
    @test resolve(p, (n_rooms = 3,)) === 3

    p2 = Param{Int}(7)
    @test p2 isa Param{Int}
    @test resolve(p2, NamedTuple()) === 7
end

@testset "Param — mode flip via field mutation" begin
    p = Param(0.1)
    @test resolve(p, NamedTuple()) ≈ 0.1
    p.val = :ξ
    @test is_swept(p)
    @test resolve(p, (ξ = 0.9,)) ≈ 0.9
    # And back:
    p.val = 0.3
    @test !is_swept(p)
    @test resolve(p, NamedTuple()) ≈ 0.3
end

@testset "Param — resolve is type-stable" begin
    p_lit = Param(0.25)
    p_sym = Param(:foo)
    env = (foo = 0.5,)
    @inferred resolve(p_lit, env)
    @inferred resolve(p_sym, env)
end
