@testset "Testing Bruteforce Search" begin
    dataset = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, dataset_path)
    queries = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, query_path)

    ids = GraphANN.bruteforce_search(queries, dataset)

    # NOTE: no index-0 to index-1 translation because `bruteforce_search` automatically
    # converts to index-0>
    gt = GraphANN.load_vecs(groundtruth_path)

    # Unfortunately, ordering can be a little tricky since vectors with the same distance
    # can be swapped.
    #
    # The strategy here is to find all the entries that are mismatched and ensure that
    # the error is just a swap in the ranking.
    #
    # This `findall` construct returns a bunch of CartesianIndices.
    # We take these by groups of two and make sure subsequenct indices differ by only 1
    # in the column dimension (first dimension)
    mismatches = findall(ids .!= gt)
    rank_swap = CartesianIndex(1,0)
    for (a, b) in Iterators.partition(mismatches, 2)
        @test b - a == rank_swap

        # Test that the calculated distances are also the same.
        _, acol = Tuple(a)
        query = queries[acol]

        # Need to add 1 to convert from the 0-based indexing to Julia's 1-based indexing.
        @test ==(
            GraphANN.distance(query, dataset[ids[a] + 1]),
            GraphANN.distance(query, dataset[ids[b] + 1]),
        )

        @test ids[a] == gt[b]
        @test gt[b] == ids[a]
    end
end
