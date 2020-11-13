@testset "Testing Pruner" begin
    # Make this definition for convenience
    Pruner = GraphANN.Pruner

    x = [3,2,1]
    pruner = Pruner{eltype(x)}()

    @test length(pruner) == 0

    # Add all the items in `x` to the pruner.
    GraphANN.initialize!(pruner, x)
    @test length(pruner) == 3
    @test GraphANN.ispruned(pruner, 1) == false
    @test GraphANN.ispruned(pruner, 2) == false
    @test GraphANN.ispruned(pruner, 3) == false
    @test collect(pruner) == x

    # Does sorting work?
    sort!(pruner)
    @test collect(pruner) == sort(x)

    # Prune out the middle element
    GraphANN.prune!(pruner, 2)
    @test GraphANN.ispruned(pruner, 1) == false
    @test GraphANN.ispruned(pruner, 2) == true
    @test GraphANN.ispruned(pruner, 3) == false
    @test collect(pruner) == [1,3]

    # Prune out other items
    GraphANN.prune!(pruner, 3)
    @test GraphANN.ispruned(pruner, 1) == false
    @test GraphANN.ispruned(pruner, 2) == true
    @test GraphANN.ispruned(pruner, 3) == true
    @test collect(pruner) == [1]

    GraphANN.prune!(pruner, 1)
    @test GraphANN.ispruned(pruner, 1) == true
    @test GraphANN.ispruned(pruner, 2) == true
    @test GraphANN.ispruned(pruner, 3) == true
    @test collect(pruner) == []

    # Try again, but now use the filter function.
    pruner = GraphANN.Pruner{Int}()
    GraphANN.initialize!(pruner, 1:100)
    GraphANN.prune!(isodd, pruner)
    @test collect(pruner) == 2:2:100
end

@testset "Testing Index" begin
    datadir = joinpath(dirname(@__DIR__), "..", "data")
    dataset_path = joinpath(datadir, "siftsmall_base.fvecs")
    query_path = joinpath(datadir, "siftsmall_query.fvecs")
    groundtruth_path = joinpath(datadir, "siftsmall_groundtruth.ivecs")

    # Load the dataset into memory
    # For now, we have to resort to a little dance to convert the in-memory representation
    # to a collection of `euclidean` points.
    #
    # Eventually, we'll have support for directly loading into a vector of `points`.
    dataset = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, dataset_path)

    alpha = 1.2
    max_degree = 70
    window_size = 50
    parameters = GraphANN.GraphParameters(alpha, max_degree, window_size, 0.8)

    meta = GraphANN.generate_index(dataset, parameters)
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
    meta = GraphANN.MetaGraph(g, dataset)
    start = GraphANN.medioid(dataset)

    ids = GraphANN.searchall(algo, meta, start, queries; num_neighbors = 100)
    recalls = GraphANN.recall(ground_truth, ids)
    @test mean(recalls) >= 0.99
end
