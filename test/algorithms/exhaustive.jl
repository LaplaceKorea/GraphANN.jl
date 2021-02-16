@testset "Testing Exhaustive Search" begin
    @testset "Testing Exhaustive Utilities" begin
        # the `_num_neighbors` function should differentiate vectors and matrices.
        x = [1,2,3]
        @test GraphANN.Algorithms._num_neighbors(x) == 1
        for i in 1:10
            y = rand(Float32, rand(1:10), rand(1:10))
            @test GraphANN.Algorithms._num_neighbors(y) == size(y, 1)
        end

        # Test `threadlocal_wrap`.
        x = [1,2,3]
        y = GraphANN.Algorithms.threadlocal_wrap(GraphANN.dynamic_thread, x)
        @test isa(y, GraphANN.ThreadLocal{typeof(x)})

        # should be identity function and return the exact same object.
        y = GraphANN.Algorithms.threadlocal_wrap(GraphANN.single_thread, x)
        @test y === x

        # # -- exhaustive_threadlocal
        # # single threaded
        # num_neighbors = 10
        # groupsize = 16
        # x = GraphANN.Algorithms.exhaustive_threadlocal(
        #     GraphANN.single_thread,
        #     UInt64,
        #     Float32,
        #     num_neighbors,
        #     groupsize,
        # )
        # @test isa(x, Vector{GraphANN.BoundedMaxHeap{GraphANN.Neighbor{UInt64, Float32}}})
        # @test length(x) == groupsize
        # @test all(i -> GraphANN._Base.getbound(i) == num_neighbors, x)

        # # multi-threaded
        # x = GraphANN.Algorithms.exhaustive_threadlocal(
        #     GraphANN.dynamic_thread,
        #     UInt32,
        #     Int32,
        #     num_neighbors,
        #     groupsize,
        # )

        # @test isa(x, GraphANN.ThreadLocal)
        # # all thread local objects should
        # storage = GraphANN.getall(x)
        # @test all(i -> length(i) == groupsize, storage)
        # for y in storage
        #     @test all(i -> GraphANN._Base.getbound(i) == num_neighbors, y)
        # end

        # -- _populate!, _set!, and commit!

        # since bounded max heaps keep the smalles elements, we'll add Neighbors starting
        # from larger distances to smaller distances.
        #
        # We know that the `num_neighbors` smallest Neighbors will remain at the end, so
        # we can easily test the expected results.
        num_neighbors = 10
        heap = GraphANN.BoundedMaxHeap{GraphANN.Neighbor{UInt32,Int32}}(num_neighbors)
        for i in 20:-1:1
            push!(heap, GraphANN.Neighbor{UInt32,Int32}(i, Int32(i)))
        end

        # populate a vector with all neighbors.
        gt = Vector{GraphANN.Neighbor{UInt32,Int32}}(undef, num_neighbors)
        GraphANN.Algorithms._populate!(gt, heap)
        for (i, n) in enumerate(gt)
            @test n == GraphANN.Neighbor{UInt32,Int32}(i, Int32(i))
        end

        # do this again, but now with just an integer destination array
        empty!(heap)
        for i in 20:-1:1
            push!(heap, GraphANN.Neighbor{UInt32,Int32}(i, Int32(i)))
        end
        gt = Vector{UInt32}(undef, num_neighbors)
        GraphANN.Algorithms._populate!(gt, heap)
        for (i, n) in enumerate(gt)
            @test n == i
        end

        # test `_set!`
        y = GraphANN.Neighbor{UInt32,Int32}(10, Int32(100))
        gt = Vector{GraphANN.Neighbor{UInt32,Int32}}(undef, 1)
        GraphANN.Algorithms._set!(gt, y, 1)
        @test gt[1] == y

        gt = Vector{UInt32}(undef, 1)
        GraphANN.Algorithms._set!(gt, y, 1)
        @test gt[1] == GraphANN.getid(y)

        # finally, test all flavors of `_commit!`
        heaps = [
            GraphANN.BoundedMaxHeap{GraphANN.Neighbor{UInt64,Float64}}(10),
            GraphANN.BoundedMaxHeap{GraphANN.Neighbor{UInt64,Float64}}(10),
        ]
        # do the same trick for populating the heaps in such a way that we will know what
        # the actual result will be.
        function populate!(heaps)
            foreach(empty!, heaps)
            for i in 20:-1:1
                push!(heaps[1], GraphANN.Neighbor{UInt64,Float64}(i, Float64(i)))
            end
            for i in 120:-1:101
                push!(heaps[2], GraphANN.Neighbor{UInt64,Float64}(i, Float64(i)))
            end
        end
        function check(gt::Matrix{<:Number})
            for i in 1:10
                @test gt[i,1] == i
                @test gt[i,2] == i + 100
            end
        end
        function check(gt::Matrix{GraphANN.Neighbor{I,D}}) where {I,D}
            for i in 1:10
                @test gt[i,1] == GraphANN.Neighbor{I,D}(i, D(i))
                @test gt[i,2] == GraphANN.Neighbor{I,D}(i + 100, D(i + 100))
            end
        end

        # neighbor mode
        populate!(heaps)
        gt = Matrix{GraphANN.Neighbor{UInt64,Float64}}(undef, 10, 2)
        GraphANN.Algorithms._commit!(gt, heaps, 1:2)
        check(gt)

        # integer mode
        populate!(heaps)
        gt = Matrix{UInt64}(undef, 10, 2)
        GraphANN.Algorithms._commit!(gt, heaps, 1:2)
        check(gt)

        # _commit! vector mode
        heaps = [
            GraphANN.BoundedMaxHeap{GraphANN.Neighbor{UInt64,Float64}}(1),
            GraphANN.BoundedMaxHeap{GraphANN.Neighbor{UInt64,Float64}}(1),
        ]
        populate!(heaps)
        gt = Vector{GraphANN.Neighbor{UInt64,Float64}}(undef, 2)
        GraphANN.Algorithms._commit!(gt, heaps, 1:2)
        @test gt[1] == GraphANN.Neighbor{UInt64,Float64}(1, Float64(1))
        @test gt[2] == GraphANN.Neighbor{UInt64,Float64}(101, Float64(101))

        populate!(heaps)
        gt = Vector{UInt64}(undef, 2)
        GraphANN.Algorithms._commit!(gt, heaps, 1:2)
        @test gt[1] == 1
        @test gt[2] == 101
    end

    function compare_ids(ids)
        # NOTE: no index-0 to index-1 translation because `exhaustive_search` automatically
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
        metric = GraphANN.Euclidean()
        for (a, b) in Iterators.partition(mismatches, 2)
            @test b - a == rank_swap

            # Test that the calculated distances are also the same.
            _, acol = Tuple(a)
            query = queries[acol]

            # Need to add 1 to convert from the 0-based indexing to Julia's 1-based indexing.
            @test ==(
                GraphANN.evaluate(metric, query, dataset[ids[a] + 1]),
                GraphANN.evaluate(metric, query, dataset[ids[b] + 1]),
            )

            @test ids[a] == gt[b]
            @test gt[b] == ids[a]
        end
    end

    dataset = GraphANN.load_vecs(SVector{128,Float32}, dataset_path)
    queries = GraphANN.load_vecs(SVector{128,Float32}, query_path)

    # Note - `exhaustive_search` already returns nearest neighbors in index-0, so no need
    # to convert the ground truth to index-1.
    gt = GraphANN.load_vecs(groundtruth_path)

    # Using Float32
    ids = GraphANN.exhaustive_search(queries, dataset; metric = GraphANN.Euclidean())
    compare_ids(ids)

    # Using UInt8
    queries_u8 = [map(UInt8, i) for i in queries]
    dataset_u8 = [map(UInt8, i) for i in dataset]
    ids = GraphANN.exhaustive_search(
        queries_u8,
        dataset_u8;
        metric = GraphANN.Euclidean(),
    )
    compare_ids(ids)
end
