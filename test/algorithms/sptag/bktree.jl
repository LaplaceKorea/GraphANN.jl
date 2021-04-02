@testset "Testing BKTree" begin

@testset "Utility Functions" begin
    # -- doubleview
    x = 1:10
    y = [5,4,3]
    z = [1,3]
    vv = GraphANN.Algorithms.doubleview(x, y, z)
    @test length(vv) == length(z)
    @test vv[1] == x[y[z[1]]]
    @test vv[2] == x[y[z[2]]]

    # -- shrink
    x = collect(1:10)
    GraphANN.Algorithms.shrink!(x, 5)
    @test x == 1:5
    GraphANN.Algorithms.shrink!(x, 0)
    @test x == 1:5

    x = 1:10
    y = GraphANN.Algorithms.shrink!(x, 5)
    @test y === 1:5
    z = GraphANN.Algorithms.shrink!(y, 0)
    @test z === 1:5

    # -- move_to_end
    x = collect(1:10)
    GraphANN.Algorithms.move_to_end!(x, [1,2,3])
    @test sort(x) == 1:10
    @test sort(view(x, 8:10)) == [1,2,3]

    # Trickier case - moving to end when there are indexes already there.
    x = collect(1:10)
    GraphANN.Algorithms.move_to_end!(x, [1, 10, 9])
    @test sort(x) == 1:10
    @test sort(view(x, 8:10)) == [1, 9, 10]

    # -- findfirstfrom
    x = zeros(Int64, 100)
    for i in [1, 10, 50, 75]
        x[i] = 1
    end
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 1) == 1
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 2) == 10
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 3) == 10
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 9) == 10
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 10) == 10
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 11) == 50
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 51) == 75
    @test GraphANN.Algorithms.findfirstfrom(!iszero, x, 76) == length(x) + 1

    # -- Dual
    x = collect(1:10)
    y = reverse(x)
    v = GraphANN.Algorithms.Dual(x, y)
    @test size(v) == size(x)
    @test size(v) == size(y)

    # Does indexing work as expected?
    for i in 1:length(v)
        @test v[i] === (x[i], y[i])
    end

    # Try sorting. If we sort by "y", then we'd expect "y" to become ordered and
    # for "x" to become reversed.
    @test issorted(x)
    @test issorted(y; rev = true)
    sort!(v; by = last)
    @test issorted(x; rev = true)
    @test issorted(y)

    # We've implicitly tested "setindex!" via sorting above, but lets have an explicit
    # test for it too.
    v[1] = (100, -100)
    @test x[1] == 100
    @test y[1] == -100

    # Test for argument error if passing different length arrays.
    x = collect(1:10)
    y = collect(1:11)
    @test_throws ArgumentError GraphANN.Algorithms.Dual(x, y)
    @test_throws ArgumentError GraphANN.Algorithms.Dual(y, x)
end

@testset "BKTree Building" begin
    data = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, dataset_path)

    # First, make sure the stack builder is working correctly.
    permutation = UInt32.(eachindex(data))::Vector{UInt32}

    # Get pretty specific with the element type of the stack vector since it's relied
    # on somewhat heavily in the code.
    for executor in (GraphANN.single_thread, GraphANN.dynamic_thread)
        builder = GraphANN._Trees.TreeBuilder{UInt32}(length(data))
        kmeans_runner = GraphANN.KMeansRunner(data, executor)
        exhaustive_runner = GraphANN.Algorithms.ExhaustiveRunner(
            length(data),
            one,
            UInt32;
            executor = executor,
            costtype = Float32,
        )

        # Get pretty specific with the element type of the stack vector since it's relied
        # on somewhat heavily in the code.
        instack = [(parent = 0, range = 1:length(data))]
        outstack = Vector{eltype(instack)}()

        sort!(permutation)
        f = x -> push!(outstack, x)
        fanout = 4
        leafsize = 8
        GraphANN.Algorithms.process_stack!(
            f,
            instack,
            builder,
            data,
            permutation,
            kmeans_runner,
            exhaustive_runner,
            fanout,
            leafsize,
        )

        # Is our permutation vector still a permutation?
        @test sort(permutation) == eachindex(data)

        # Make sure all NamedTuples pushed to the outstack have a length less than the
        # provided leafsize.
        #
        # Also, get the last valid index in the builder.
        # The parent indices for all leaves in the outstack must be less than this last valid
        # index.
        tree_max_index = builder.last_valid_index
        for nt in outstack
            @test length(nt.range) <= leafsize
            @test 0 < nt.parent <= tree_max_index
        end

        # Make sure all nodes are accounted for.
        s = sum(length, nt.range for nt in outstack)
        @test s + tree_max_index == length(data)
    end

    # Now - test the whole pipeline.
    fanout = 4
    leafsize = 16
    # Make the stack split point small enough so we execute both pass types.
    stacksplit = div(length(data), 10)
    tree = GraphANN.Algorithms.bktree(
        data;
        fanout = fanout,
        leafsize = leafsize,
        stacksplit = stacksplit,
        idtype = UInt32,
    )

    # Check properties of the nodes.
    nodes = tree.nodes
    max_children = 0
    isleaf = GraphANN._Trees.isleaf
    for node in Iterators.filter(!isleaf, nodes)
        # If all the children are leaves, than this node is one above a leaf so can
        # have up to `leafsize` children.
        #
        # Otherwise, it can only have a maximum of `fanout` children.
        children = GraphANN._Trees.children(tree, node)
        if all(isleaf, children)
            @test length(children) <= leafsize
            max_children = max(max_children, length(children))
        else
            @test length(children) <= fanout
        end
    end

    # At least one leaf-parent should have more children than the fanout.
    # This ensures that the switch is happening properly.
    @test fanout < max_children <= leafsize

    # The tree should be validated before it is returned, but it doesn't hurt to check
    # again in case that pass gets removed in the future for some reason.
    @test GraphANN._Trees.validate(tree)

    ## Finally, lets test the quality of the tree by performing a search and computing
    # the recall

    num_neighbors = 5
    queries = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, query_path)
    groundtruth = GraphANN.load_vecs(groundtruth_path) .+ 1

    nearest_neighbors = Vector{Vector{UInt32}}()
    numleaves = count(isleaf, tree.nodes)
    maxleaves = div(numleaves, 10)

    runner = GraphANN.Algorithms.SPTAGRunner(num_neighbors; costtype = Float32)
    for query in queries
        GraphANN.Algorithms.init!(runner, tree, data, query; metric = GraphANN.Euclidean())
        leaves_seen = GraphANN.Algorithms.search(runner, tree, data, query, maxleaves)

        # Number of items pushed to the `graph_queue` should be greater than `maxleaves`.
        @test length(runner.graph_queue) >= maxleaves
        neighbors = DataStructures.extract_all!(runner.graph_queue)
        @test allunique(neighbors)

        # Make sure we have the correct number of leaf nodes.
        @test maxleaves == leaves_seen
        push!(nearest_neighbors, GraphANN.getid.(view(neighbors, 1:num_neighbors)))
    end
    nearest_neighbors = reduce(hcat, nearest_neighbors)

    recall = mean(GraphANN.recall(groundtruth, nearest_neighbors))

    # Should be able to achieve ~96% 5 recall at 5 by only visiting 10% of the leaves.
    @test recall > 0.96
end

end
