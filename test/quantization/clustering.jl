@testset "Testing CurrentMinimum" begin
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

@testset "Testing PackedCentroids" begin
    x = GraphANN._Quantization.PackedCentroids{GraphANN.Euclidean{4,Float32}}(32, 16)

    # With vector sizes of 4, we can pack 4 together in a cache line.
    # Thus, the size of the resulting packed representation should be one-quarter the size.
    @test size(x) == (div(32, 4), 16)
    @test iszero(get(x, 1, 1, 1))
    @test all(iszero(x.lengths))

    # Lets try adding some items.
    e = GraphANN.Euclidean(Float32.((1.0, 0.0, 0.0, 0.0)))
    push!(x, e, 1, 1)
    @test x.lengths[1] == 1
    @test x.lengths[2] == 0
    @test get(x, 1, 1, 1) === e
    @test iszero(get(x, 1, 1, 2))
    # is the correct mask set?
    m = x.masks[1,1]
    @test m === SIMD.Vec(true, false, false, false)
    # Sanity check
    @test x.masks[2,1] === SIMD.Vec(false, false, false, false)

    # Lets add another one!
    f = GraphANN.Euclidean(Float32.((0.0, 1.0, 0.0, 0.0)))
    push!(x, f, 1, 1)
    @test x.lengths[1] == 2
    @test get(x, 1, 1, 1) === e
    @test get(x, 1, 1, 2)=== f

    @test x.masks[1,1] === SIMD.Vec(true, false, false, false)
    @test x.masks[1,2] === SIMD.Vec(true, false, false, false)
    @test x.masks[2,1] === SIMD.Vec(false, false, false, false)
    @test x.masks[2,2] === SIMD.Vec(false, false, false, false)

    # Try adding to another lane in the same group
    push!(x, f, 2, 1)
    @test get(x, 2, 1, 1) === f
    @test iszero(get(x, 3, 1, 1))

    push!(x, e, 2, 1)
    @test get(x, 2, 1, 2) === e
    @test x.masks[1, 1] === SIMD.Vec(true, true, false, false)

    # Now, add to another offset in another group
    push!(x, e, 4, 5)
    @test get(x, 4, 5, 1) === e
    @test x.masks[5, 1] === SIMD.Vec(false, false, false, true)
end
