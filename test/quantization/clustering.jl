@testset "Testing Clustering" begin
    # Test `CurrentMinimum`
    x = GraphANN._Quantization.CurrentMinimum{2,Float32}()
    @test length(x.distance) == 2
    @test length(x.index) == 2

    d = SIMD.Vec(Float32.((10, 20)))
    x = GraphANN._Quantization.update(x, d, 1)
    @test Tuple(x.distance) == (10, 20)
    @test Tuple(x.index) == (1, 1)

    # Now, only update one of the entries
    d = SIMD.Vec(Float32.((5, 50)))
    x = GraphANN._Quantization.update(x, d, 2)
    @test Tuple(x.distance) == (5, 20)
    @test Tuple(x.index) == (2, 1)

    # Now, update the other one
    d = SIMD.Vec(Float32.((50, 2)))
    x = GraphANN._Quantization.update(x, d, 3)
    @test Tuple(x.distance) == (5, 2)
    @test Tuple(x.index) == (2, 3)
end
