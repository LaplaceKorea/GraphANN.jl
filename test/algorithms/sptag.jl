@testset "SPTAG Search" begin
    # -- "getid" for "Neighbor{<:TreeNode}"
    TreeNode = GraphANN._Trees.TreeNode
    node = TreeNode{UInt32}(1,2,3)
    neighbor = Neighbor{TreeNode{UInt32},Float32}(node, Float32(1.0))

    @test GraphANN.getid(neighbor) === UInt32(1)
    @test GraphANN.Algorithms.getnode(neighbor) === node
end

struct SPTAGComparison
    maxcheck::Int64
    expected_recall::Float64
end
SPTAGComparison(x::Tuple) = SPTAGComparison(x...)
approx_or_greater(x, y) = (x ≈ y) || (x > y)

@testset "Query Comparision" begin
    comparisons = SPTAGComparison.([
        (10,  0.7),
        (20,  0.7),
        (30,  0.7),
        (40,  0.7),
        (50,  0.7),
        (60,  0.7),
        (70,  0.79),
        (80,  0.8),
        (90,  0.8),
        (100, 0.84),
        (200, 0.93),
        (300, 0.97),
        (400, 0.99),
    ])

    data = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, dataset_path)
    queries = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, query_path)
    groundtruth = GraphANN.load_vecs(groundtruth_path) .+ 1

    tree = GraphANN._IO.load_bktree(sptag_tree).tree
    graph = GraphANN._IO.load_graph(GraphANN._IO.SPTAG(), sptag_index)

    index = GraphANN.SPTAGIndex(graph, data, tree)
    algo = GraphANN.SPTAGRunner(1; costtype = GraphANN.costtype(GraphANN.Euclidean(), data))

    for (i, comparison) in enumerate(comparisons)
        # Dividing the maximum check number by 64 is how the SPTAG code derives their
        # value for the propagation bound.
        maxcheck = comparison.maxcheck
        propagation_limit = GraphANN.cdiv(maxcheck, 64)

        ids = GraphANN.search(
            algo,
            index,
            queries;
            maxcheck = maxcheck,
            propagation_limit = propagation_limit,
        )

        our_recall = mean(GraphANN.recall(groundtruth, ids))
        @test approx_or_greater(our_recall, comparison.expected_recall)
    end
end
