function map_then_return(dir, expected_path)
    @test !ispath(expected_path)
    A = GraphANN.pmmap(Float32, dir, 100)
    @test isa(A, Vector{Float32})
    @test length(A) == 100
    @test ispath(expected_path)

    return nothing
end

@testset "Testing PM" begin
    dir = @__DIR__
    expected_path = joinpath(dir, join((GraphANN.PM.mmap_prefix(), 0)))
    ispath(expected_path) && rm(expected_path)

    map_then_return(dir, expected_path)

    # Invoke GC - make sure the generated file gets cleaned up.
    GC.gc()
    @test !ispath(expected_path)
end
