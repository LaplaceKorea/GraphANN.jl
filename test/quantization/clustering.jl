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

@testset "End to End Clustering" begin
    # Hoist some things for clarity.
    # -- types
    Euclidean = GraphANN.Euclidean
    LazyArrayWrap = GraphANN._Points.LazyArrayWrap
    PackedCentroids = GraphANN._Quantization.PackedCentroids

    # -- functions
    _packed_type = GraphANN._Points.packed_type
    choose_centroids = GraphANN._Quantization.choose_centroids
    refine = GraphANN._Quantization.refine
    computecosts! = GraphANN._Quantization.computecosts!
    lloyds = GraphANN._Quantization.lloyds

    # As always - load up our test dataset.
    # Also, make sure the code paths work the same whether we are using `Float32` or
    # `UInt8`
    data_f32 = GraphANN.load_vecs(Euclidean{128,Float32}, dataset_path)
    data_u8 = map(x -> convert(Euclidean{128,UInt8}, x), data_f32)
    alldata = (data_f32, data_u8)

    # Range over partition size and number of centroids
    # TODO: Get 2 working ...
    partition_sizes = [4, 8]
    num_centroids_range = [256, 512]

    # First order of business - make sure that our process of selecting initial centroids
    # is better than choosing initial centroids randomly.
    for data in alldata, partition_size in partition_sizes, num_centroids in num_centroids_range
        # As is the theme with all this clustering stuff, hoist up some useful named values.
        num_partitions = div(128, partition_size)
        T = eltype(eltype(data))
        centroid_type = Euclidean{partition_size, T}
        packed_type = _packed_type(centroid_type)
        costtype = GraphANN.costtype(eltype(data))

        # TODO: There has to be a better way of getting this type ...

        # Display what we're currently working so people watching the test process know
        # what's going on.
        @show T partition_size num_centroids
        println()

        # Initial centroid selection, results in more centroids than asked for.
        # Need to refine the centroids to actually get the corret number.
        centroids = choose_centroids(data, partition_size, num_centroids)
        refined_centroids = refine(centroids, data, num_centroids)

        # Now that we have the refined centroids, compute the initial cost of this
        # clustering.
        costs = zeros(cost_type, num_partitions, length(data))
        packed_centroids = PackedCentroids{packed_type}(refined_centroids)
        data_wrapped = LazyArrayWrap{packed_type}(data)
        computecosts!(costs, packed_centroids, data_wrapped)

        costs_chosen_sum = sum.(eachrow(costs))

        # Now, randomly select centroids - we should always do better.
        data_partitions = LazyArrayWrap{centroid_type}(data)
        centroids_random = [
            rand(view(data_partitions, i, :)) for _ in 1:num_centroids, i in 1:num_partitions
        ]
        packed_centroids = PackedCentroids{packed_type}(centroids_random)
        computecosts!(costs, packed_centroids, data_wrapped)
        costs_random_sum = sum.(eachrow(costs))

        # As said above - the residuals for each partition when we use the kmeans∥
        # algorithm should always be less than if we just chose the centroids at random.
        #
        # Otherwise, out implementation of the algorithm sucks!
        for (a, b) in zip(costs_chosen_sum, costs_random_sum)
            @test a < b
        end

        # Now that we know we have a pretty good initial clustering, run Lloyd's algorithm!
        # Since lloyd's algorithm monotonically decreases the residuals - the final
        # results must be strictly better than the refined centroids.
        final_centroids = lloyds(refined_centroids, data; num_iterations = 10)
        new_packed_type = _packed_type(eltype(final_centroids))
        new_data_wrapped = LazyArrayWrap{new_packed_type}(data)

        costs = zeros(
            GraphANN._Points.cost_type(eltype(final_centroids)),
            num_partitions,
            length(data),
        )
        computecosts!(
            costs,
            PackedCentroids{new_packed_type}(final_centroids),
            new_data_wrapped,
        )
        costs_final_sum = sum.(eachrow(costs))
        for (a, b) in zip(costs_final_sum, costs_chosen_sum)
            @test a < b
        end
    end
end
