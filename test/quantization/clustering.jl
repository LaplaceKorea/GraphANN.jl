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
    E = GraphANN.Euclidean{4,Float32}
    V = SIMD.Vec{16,Float32}
    x = GraphANN._Quantization.PackedCentroids{E,V}(32, 16)

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

@testset "Misc Clustering Tests" begin
    maybe_widen = GraphANN._Quantization.maybe_widen
    @test maybe_widen(UInt8)  == widen(UInt8)
    @test maybe_widen(UInt16) == widen(UInt16)
    @test maybe_widen(UInt32) == widen(UInt32)
    @test maybe_widen(UInt64) == UInt64

    @test maybe_widen(Float32) == Float64
    @test maybe_widen(Float64) == Float64

    @test maybe_widen(Int32) == Int64
    @test maybe_widen(Int64) == Int64
end

# @testset "End to End Clustering" begin
#     # As always - load up our test dataset.
#     # Also, make sure the code paths work the same whether we are using `Float32` or
#     # `UInt8`
#     data_f32 = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, dataset_path)
#     data_u8 = map(x -> convert(GraphANN.Euclidean{128,UInt8}, x), data)
#     alldata = (data_f32, data_u8)
#
#     # Range over partition size and number of centroids
#     # TODO: Get 2 working ...
#     partition_sizes = [4, 8, 16]
#     num_centroids_range = [256, 512, 1024]
#
#     # First order of business - make sure that our process of selecting initial centroids
#     # is better than choosing initial centroids randomly.
#     for data in alldata, partition_size in partition_sizes, num_centroids in num_centroids_range
#         # Initial centroid selection, results in more centroids than asked for.
#         # Need to refine the centroids to actually get the corret number.
#         centroids = GraphANN._Quantization.choose_centroids(
#             data,
#             partition_size,
#             num_centroids,
#         )
#
#         refined_centroids = GraphANN._Quantization.refine(
#             centroids,
#             data,
#             num_centroids,
#         )
#
#         # Now that we have the refined centroids, compute the initial cost of this
#         # clustering.
#
#     end
# end
