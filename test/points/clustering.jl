function test_clustering(data)
    # Use 128 centroids
    # The main idea for a test is that the cost of the clustering should:
    #
    # (1) Always decrease as we apply refinements
    # (2) Be lower than the the cost of using the origin as a single cluster centroid.
    num_centroids = 128
    centroids = GraphANN._Quantization.choose_centroids(
        data,
        num_centroids;
        num_iterations = 10,
        oversample = 10,
    )
    @test length(centroids) == num_centroids
    costs = zeros(length(data))

    # Remeber - the computed cost is really the square of the cost since that's easier to
    # calculate and the square-root function is monotonic.
    initial_cost = sum(GraphANN._Quantization.compute_cost!(costs, data, centroids))
    origin_cost = sum(sum, data)
    @test sqrt(initial_cost) < origin_cost

    # Refine using Lloyd's algorithm
    centroids = GraphANN._Quantization.lloyds(
        centroids,
        data;
        max_iterations = 50,
        tol = 1E-5,
        batchsize = 1024,
    )

    cost = sum(GraphANN._Quantization.compute_cost!(costs, data, centroids))
    @test cost < initial_cost
    @test sqrt(cost) < origin_cost
    return cost
end

@testset "Testing Clustering" begin
    data = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, dataset_path)

    # Try clustering with both Float32 and UInt8 data types.
    cost_f32 = test_clustering(data)
    cost_u8 = test_clustering(map(i -> map(UInt8, i), data))
    # Relative error between the computed costs should be pretty low
    rel_error = abs(cost_f32 - cost_u8) / cost_f32
    @test rel_error < 1E-2
end
