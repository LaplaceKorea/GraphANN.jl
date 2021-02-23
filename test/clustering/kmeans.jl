@testset "Testing KMeans" begin

@testset "Utils" begin
    x = GraphANN.Neighbor{Int64}(1, 100)
    # Resetting should return a maxed out neighbor
    y = GraphANN._Clustering.reset(x)
    @test isa(y, GraphANN.Neighbor{Int64,Int64})
    # Put field checks here to make sure that if the definition of Neighbor() ever changes
    # then this will fail and we can update the code.
    @test y === GraphANN.Neighbor{Int64,Int64}()
    @test GraphANN.getid(y) === 0
    @test GraphANN.getdistance(y) === typemax(Int64)

    # Test `vupdate`
    w = @SVector fill(Float32(1), 16)
    x = @SVector fill(Float32(6), 16)
    y = @SVector fill(Float32(5), 16)
    z = @SVector fill(Float32(7), 16)

    minimum = GraphANN.Neighbor{UInt32,Float32}()
    # Should update
    minimum = GraphANN._Clustering.vupdate(minimum, w, x, 1; metric = GraphANN.Euclidean())
    @test minimum === GraphANN.Neighbor{UInt32,Float32}(1, Float32(16 * (6 - 1)^2))

    # Should update again
    minimum = GraphANN._Clustering.vupdate(minimum, w, y, 2, metric = GraphANN.Euclidean())
    @test minimum === GraphANN.Neighbor{UInt32,Float32}(2, Float32(16 * (5 - 1)^2))

    # Should NOT update this time.
    minimum = GraphANN._Clustering.vupdate(minimum, w, z, 3, metric = GraphANN.Euclidean())
    @test minimum === GraphANN.Neighbor{UInt32,Float32}(2, Float32(16 * (5 - 1)^2))

    # Finally, test that `findnearest!` works.
    # All points in the dataset should be nearest to themselves.
    data = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, dataset_path)
    for i in eachindex(data)
        minimum = GraphANN.Neighbor{UInt32,Float32}()
        new_minimum = GraphANN._Clustering.findnearest!(
            data[i],
            data,
            minimum;
            metric = GraphANN.Euclidean(),
        )
        @test GraphANN.getid(new_minimum) == i
        @test GraphANN.getdistance(new_minimum) == 0
    end
end

function clustering_cost(centroids, dataset)
    cost = zero(Float64)
    for datum in dataset
        minimum = GraphANN._Clustering.findnearest!(
            datum,
            centroids,
            GraphANN.Neighbor{UInt32,Float32}();
            metric = GraphANN.Euclidean(),
        )

        cost += GraphANN.getdistance(minimum)
    end
    return cost
end

@testset "Runner" begin
    data_f32 = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, dataset_path)
    data_u8 = GraphANN.toeltype(UInt8, data_f32)

    num_centroids = 16

    # First - get a baseline cost for if we randomly choose centroids.
    Random.seed!(1234)
    centroids = rand(data_f32, num_centroids)
    baseline_cost = clustering_cost(centroids, data_f32)

    # Try single threaded versions first.
    runner = GraphANN.KMeansRunner(data_f32, GraphANN.single_thread)
    @test GraphANN._Clustering.is_single_thread(runner)
    centroids = GraphANN.kmeans(data_f32, runner, num_centroids)
    @test isa(centroids, Vector{GraphANN.SVector{128,Float32}})
    clustered_cost = clustering_cost(centroids, data_f32)

    st_runtime = @elapsed GraphANN.kmeans(data_f32, runner, num_centroids)

    # Cost ratio determined by running this code and checking the ratio.
    # It at least serves as a baseline to detect against regression.
    @test clustered_cost < 0.6 * baseline_cost

    # Try clustering using the UInt8 dataset
    centroids = GraphANN.kmeans(data_u8, runner, num_centroids)
    clustered_cost = clustering_cost(centroids, data_u8)
    @test clustered_cost < 0.6 * baseline_cost

    # Now, try multithreading!
    runner = GraphANN.KMeansRunner(data_u8, GraphANN.dynamic_thread)
    @test GraphANN._Clustering.is_single_thread(runner) == false

    centroids = GraphANN.kmeans(data_f32, runner, num_centroids)
    clustered_cost = clustering_cost(centroids, data_f32)
    @test clustered_cost < 0.6 * baseline_cost
    mt_runtime = @elapsed GraphANN.kmeans(data_f32, runner, num_centroids)

    # Again, this ratio of runtime is determined heuristically.
    # With at least 2 threads, we should get close to a 2x speedup.
    @test Threads.nthreads() >= 2
    @test mt_runtime < 0.6 * st_runtime

    # Finally, multi-threaded UInt8
    centroids = GraphANN.kmeans(data_u8, runner, num_centroids)
    clustered_cost = clustering_cost(centroids, data_u8)
    @test clustered_cost < 0.6 * baseline_cost
end

end
