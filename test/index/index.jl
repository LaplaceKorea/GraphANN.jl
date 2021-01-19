
@testset "Testing Pruner" begin
    # Make this definition for convenience
    Pruner = GraphANN.DiskANN.Pruner

    x = [3,2,1]
    pruner = Pruner{eltype(x)}()

    @test length(pruner) == 0

    # Add all the items in `x` to the pruner.
    GraphANN.DiskANN.initialize!(pruner, x)
    @test length(pruner) == 3
    @test GraphANN.DiskANN.ispruned(pruner, 1) == false
    @test GraphANN.DiskANN.ispruned(pruner, 2) == false
    @test GraphANN.DiskANN.ispruned(pruner, 3) == false
    @test collect(pruner) == x

    # Does sorting work?
    sort!(pruner)
    @test collect(pruner) == sort(x)

    # Prune out the middle element
    GraphANN.DiskANN.prune!(pruner, 2)
    @test GraphANN.DiskANN.ispruned(pruner, 1) == false
    @test GraphANN.DiskANN.ispruned(pruner, 2) == true
    @test GraphANN.DiskANN.ispruned(pruner, 3) == false
    @test collect(pruner) == [1,3]

    # Prune out other items
    GraphANN.DiskANN.prune!(pruner, 3)
    @test GraphANN.DiskANN.ispruned(pruner, 1) == false
    @test GraphANN.DiskANN.ispruned(pruner, 2) == true
    @test GraphANN.DiskANN.ispruned(pruner, 3) == true
    @test collect(pruner) == [1]

    GraphANN.DiskANN.prune!(pruner, 1)
    @test GraphANN.DiskANN.ispruned(pruner, 1) == true
    @test GraphANN.DiskANN.ispruned(pruner, 2) == true
    @test GraphANN.DiskANN.ispruned(pruner, 3) == true
    @test collect(pruner) == []

    # Try again, but now use the filter function.
    pruner = GraphANN.DiskANN.Pruner{Int}()
    GraphANN.DiskANN.initialize!(pruner, 1:100)
    GraphANN.DiskANN.prune!(isodd, pruner)
    @test collect(pruner) == 2:2:100
end

function test_index(dataset, graph_type = GraphANN.DefaultAdjacencyList{UInt32})
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
    queries = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, query_path)

    # Need to adjust ground-truth from index-0 to index-1
    ground_truth = GraphANN.load_vecs(groundtruth_path) .+ UInt32(1)

    @test eltype(ground_truth) == UInt32
    algo = GraphANN.GreedySearch(100)
    start = GraphANN.medioid(dataset)
    start = GraphANN.StartNode(start, dataset[start])

    ids = GraphANN.searchall(algo, meta, start, queries; num_neighbors = 100)
    recalls = GraphANN.recall(ground_truth, ids)
    @test mean(recalls) >= 0.99
    return meta
end

@testset "Testing Index" begin
    # Load the dataset into memory
    dataset = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, dataset_path)::Vector{GraphANN.Euclidean{128,Float32}}
    dataset_u8 = map(x -> convert(GraphANN.Euclidean{128,UInt8}, x), dataset)

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
