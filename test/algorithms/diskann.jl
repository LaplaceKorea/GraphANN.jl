struct DiskANNComparison
    window_size::Int
    num_neighbors::Int
    expected_accuracy::Float64
end
DiskANNComparison(x::Tuple) = DiskANNComparison(x...)

# There's one test where GraphANN returns 0.9689999999999999 while DiskANN
# returns 0.9690000000000001.
#
# I mean ... come one - these are practically the same.
# SO, we make a slightly looser floating point definition for floating point
# approximate equality or greater.
approx_or_greater(a, b) = (a ≈ b) || (a > b)

# Test construction and recall on a small dataset.
function test_index(
    dataset::Vector{SVector{N,T}},
    graph_type = GraphANN.DefaultAdjacencyList{UInt32}
) where {N,T}

    alpha = 1.2
    max_degree = 70
    window_size = 50
    parameters = GraphANN.GraphParameters(;
        alpha = alpha,
        window_size = window_size,
        target_degree = max_degree,
        prune_threshold_degree = 90,
        prune_to_degree = 65,
    )

    meta = GraphANN.generate_index(dataset, parameters; graph_type = graph_type)
    g = meta.graph

    # Is the maximum degree of the generated graph within the set limit?
    @test maximum(outdegree(g)) <= max_degree
    @test is_connected(g)

    # Lets try a search
    queries = GraphANN.load_vecs(SVector{N,Float32}, query_path)
    queries = map(x -> map(T, x), queries)

    # Need to adjust ground-truth from index-0 to index-1
    ground_truth = GraphANN.load_vecs(groundtruth_path) .+ UInt32(1)

    @test eltype(ground_truth) == UInt32
    algo = GraphANN.GreedySearch(100; costtype = GraphANN.costtype(dataset), idtype = UInt32)
    start_index = GraphANN.medioid(dataset)
    start = GraphANN.StartNode(start_index, dataset[start_index])

    ids = GraphANN.searchall(algo, meta, start, queries; num_neighbors = 100)
    recalls = GraphANN.recall(ground_truth, ids)
    @test mean(recalls) >= 0.99
    return meta
end


@testset "Testing DiskANN Based Code" begin

@testset "DiskANN Search" begin
    # Test some properties of `Neighbor`
    x = GraphANN.GreedySearch(2; costtype = Float64, idtype = Int64)
    GraphANN.Algorithms.pushcandidate!(x, Neighbor(x, 1, 0.0))

    # Test `Neighbor` constructor for the `GreedySearch` type.
    @test isa(GraphANN.Neighbor(x, 1, 1.0), GraphANN.Neighbor{Int64,Float64})
    # convert id types
    @test isa(GraphANN.Neighbor(x, UInt32(1.0), 1.0), GraphANN.Neighbor{Int64,Float64})
    # no implicit conversion of distance types
    @test_throws MethodError GraphANN.Neighbor(x, 1.0, Float32(10))

    # Use `egal` for comparison in case we want to overload `==`
    @test maximum(x) === Neighbor(x, 1, 0.0)
    GraphANN.Algorithms.pushcandidate!(x, Neighbor(x, 2, 10.0))
    @test maximum(x) === Neighbor(x, 2, 10.0)

    GraphANN.Algorithms.pushcandidate!(x, Neighbor(x, 3, -10.0))
    @test maximum(x) === Neighbor(x, 2, 10.0)

    GraphANN.Algorithms.reduce!(x)
    @test maximum(x) === Neighbor(x, 1, 0.0)
end

@testset "Pruner" begin
    # Make this definition for convenience
    Pruner = GraphANN.Algorithms.Pruner

    x = [3,2,1]
    pruner = Pruner{eltype(x)}()

    @test length(pruner) == 0

    # Add all the items in `x` to the pruner.
    GraphANN.Algorithms.initialize!(pruner, x)
    @test length(pruner) == 3
    @test GraphANN.Algorithms.ispruned(pruner, 1) == false
    @test GraphANN.Algorithms.ispruned(pruner, 2) == false
    @test GraphANN.Algorithms.ispruned(pruner, 3) == false
    @test collect(pruner) == x

    # Does sorting work?
    sort!(pruner)
    @test collect(pruner) == sort(x)

    # Prune out the middle element
    GraphANN.Algorithms.prune!(pruner, 2)
    @test GraphANN.Algorithms.ispruned(pruner, 1) == false
    @test GraphANN.Algorithms.ispruned(pruner, 2) == true
    @test GraphANN.Algorithms.ispruned(pruner, 3) == false
    @test collect(pruner) == [1,3]

    # Prune out other items
    GraphANN.Algorithms.prune!(pruner, 3)
    @test GraphANN.Algorithms.ispruned(pruner, 1) == false
    @test GraphANN.Algorithms.ispruned(pruner, 2) == true
    @test GraphANN.Algorithms.ispruned(pruner, 3) == true
    @test collect(pruner) == [1]

    GraphANN.Algorithms.prune!(pruner, 1)
    @test GraphANN.Algorithms.ispruned(pruner, 1) == true
    @test GraphANN.Algorithms.ispruned(pruner, 2) == true
    @test GraphANN.Algorithms.ispruned(pruner, 3) == true
    @test collect(pruner) == []

    # Try again, but now use the filter function.
    pruner = GraphANN.Algorithms.Pruner{Int}()
    GraphANN.Algorithms.initialize!(pruner, 1:100)
    GraphANN.Algorithms.prune!(isodd, pruner)
    @test collect(pruner) == 2:2:100
end

@testset "Testing Index" begin
    # Load the dataset into memory
    dataset = GraphANN.load_vecs(SVector{128,Float32}, dataset_path)::Vector{SVector{128,Float32}}
    dataset_u8 = map(x -> GraphANN.toeltype(UInt8, x), dataset)

    # Index generation using both Float32 and UInt8
    meta = test_index(dataset)
    test_index(dataset, GraphANN.FlatAdjacencyList{UInt32})
    test_index(dataset_u8)
    test_index(dataset_u8, GraphANN.FlatAdjacencyList{UInt32})

    #####
    ##### Graph IO
    #####

    # Now that we have a functioning graph, make sure the various serialization and
    # deserialization methods work.
    mktempdir(@__DIR__) do dir
        function graphs_equal(a, b)
            @test vertices(a) == vertices(b)
            @test collect(edges(a)) == collect(edges(b))
        end

        original_save = joinpath(dir, "original")

        graphs_equal(meta.graph, meta.graph)
        GraphANN.save(original_save, meta.graph)

        # Try deserializing the original graph
        default = GraphANN.load(GraphANN.DefaultAdjacencyList{UInt32}, original_save)
        graphs_equal(default, meta.graph)

        flat = GraphANN.load(GraphANN.FlatAdjacencyList{UInt32}, original_save)
        graphs_equal(flat, meta.graph)
        graphs_equal(flat, default)

        dense = GraphANN.load(GraphANN.DenseAdjacencyList{UInt32}, original_save)
        graphs_equal(dense, meta.graph)
        graphs_equal(dense, default)
        graphs_equal(dense, flat)
    end
end

@testset "Compare With DiskANN Baseline" begin
    # Parameters taken from running the DiskANN code on the index included in the
    # "data" directory.
    #
    # This exercises both the correctness of the query algorithm as well as our computation
    # of the "recall" metric.
    comparisons = DiskANNComparison.([
        #####
        ##### SiftSmall
        #####

        (10 ,  1  , 95.0),
        (10 ,  5  , 93.4),
        (10 ,  10 , 90.9),
        (20 ,  1  , 97.0),
        (20 ,  5  , 98.0),
        (20 ,  10 , 95.2),
        (20 ,  20 , 93.9),
        (30 ,  1  , 99.0),
        (30 ,  5  , 99.0),
        (30 ,  10 , 97.5),
        (30 ,  20 , 96.95),
        (30 ,  30 , 95.8),
        (50 ,  1  , 99.0),
        (50 ,  5  , 99.2),
        (50 ,  10 , 98.9),
        (50 ,  20 , 98.8),
        (50 ,  30 , 98.5667),
        (50 ,  50 , 97.2),
        (100,  1  , 100.0),
        (100,  5  , 100.0),
        (100,  10 , 100.0),
        (100,  20 , 99.9),
        (100,  30 , 99.8),
        (100,  50 , 99.56),
        (100,  100, 98.44),
    ])

    diskann_ids = deserialize(diskann_query_ids)
    # Adjust for index-0 and index-1
    for ids in diskann_ids
        ids .+= 1
    end

    # Load in siftsmall dataset, queries, and ground truth.
    data = GraphANN.load_vecs(SVector{128,Float32}, dataset_path)
    graph = GraphANN.load_graph(GraphANN._IO.DiskANN(), diskann_index, length(data))
    meta = GraphANN.MetaGraph(graph, data)

    queries = GraphANN.load_vecs(SVector{128,Float32}, query_path)

    # Add "1" to ground truth to convert from index-0 to Julia's index-1
    ground_truth = GraphANN.load_vecs(groundtruth_path) .+ 1

    # Set up the algorithm.
    start = GraphANN.medioid(data)
    start = GraphANN.StartNode(start, data[start])

    for (i, comparison) in enumerate(comparisons)
        algo = GraphANN.GreedySearch(comparison.window_size)

        # Obtain the approximate nearest neighbors
        ids = GraphANN.searchall(
            algo,
            meta,
            start,
            queries;
            num_neighbors = comparison.num_neighbors,
        )

        recall_values = GraphANN.recall(ground_truth, ids)

        # These are based on the same algorithm, so check for approximate floating
        # point equality.
        #
        # At some point, our implementation might become better, so we'll have to adjust
        # the comparison.
        recall = 100 * mean(recall_values)
        expected = comparison.expected_accuracy
        @test recall > expected || isapprox(recall, expected; atol = 0.0001)

        # Test that the intersection between between returned ids is >99%
        # There are a couple of outliers that we need to track
        outlier_count = 0
        expected_num_outliers = 1
        for (a, b) in zip(eachcol(ids), eachcol(diskann_ids[i]))
            id_intersection = intersect(a, b)
            if length(id_intersection) < 0.99 * length(a)
                outlier_count += 1
            end
        end
        @test outlier_count <= expected_num_outliers
    end
end

end
