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

@testset "Testing Query with DiskANN" begin
    distance = GraphANN.distance

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
    data = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, dataset_path)
    graph = GraphANN.load_graph(
        GraphANN._IO.DiskANN(),
        diskann_index,
        # Sift Small has 10000 base elements.
        length(data),
    )

    meta = GraphANN.MetaGraph(graph, data)

    queries = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, query_path)

    # Add "1" to ground truth to convert from index-0 to Julia's index-1
    ground_truth = GraphANN.load_vecs(groundtruth_path) .+ 1

    # Set up the algorithm.
    start = GraphANN.medioid(data)
    start = GraphANN.StartNode(start, data[start])

    found_tail_mismatch = false
    found_swap_mismatch = false
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
        # for (a, b) in zip(eachcol(ids), eachcol(diskann_ids[i]))
        #     id_intersection = intersect(a, b)
        #     @test length(id_intersection) >= 0.95 * length(a)
        # end
    end
end
