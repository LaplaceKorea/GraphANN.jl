function _run_tpt_tests(data::AbstractVector{SVector{N,T}}) where {N,T}
    numsamples = 1000
    leafsize = 500
    tree = GraphANN.Algorithms.TPTree{4, UInt32}(
        length(data);
        leafsize = leafsize,
        numtrials = 100,
        numsamples = numsamples,
    )

    @test GraphANN.Algorithms.num_split_dimensions(tree) == 4
    @inferred GraphANN.Algorithms.getdims!(data, tree, 1:length(data))
    dims = GraphANN.Algorithms.getdims!(data, tree, 1:length(data))
    @test isa(dims, NTuple{4,Int})

    # Make sure that the maximum variances are being computed correctly.
    dataview = view(data, tree.samples)
    @test length(dataview) == numsamples
    variances = Statistics.var.(eachrow(reinterpret(reshape, T, dataview)))

    # Get the indices of the ranked variances from highest to lowest
    perm = partialsortperm(variances, 1:4; rev = true)
    @test all(dims .== perm)

    # Next, get normalized weights to maximize the varianze
    @inferred GraphANN.Algorithms.getweights!(data, tree, dims, 1:length(data))
    weights, meanval = GraphANN.Algorithms.getweights!(data, tree, dims, 1:length(data))

    # Make sure the return vector is normalized
    @test isapprox(sum(abs2, weights), 1)
    @test isa(weights, SVector{4, Float32})

    # Next - try to sort the whole data.
    # Use a callback to record all the leaf ranges.
    # Ensure that all leaves are smaller than the requested leaf size and that the
    # whole data range is covered.
    ranges = Vector{UnitRange{UInt32}}()
    f = x -> push!(ranges, x)
    single_thread = GraphANN.single_thread
    dynamic_thread = GraphANN.dynamic_thread
    GraphANN.Algorithms.partition!(f, data, tree; executor = single_thread, init = true)

    function checkranges(ranges, data, tree)
        @test !isempty(ranges)
        @test sum(length, ranges) == length(data)
        @test all(x -> length(x) <= leafsize, ranges)
        @test allunique(ranges)
        @test sort(mapreduce(collect, vcat, ranges)) == 1:length(data)

        # Check the `permutation` field. Make sure it is, in fact, still a permutation.
        permutation = tree.permutation
        @test length(permutation) == length(data)
        @test allunique(permutation)
        @test all(x -> in(x, 1:length(data)), permutation)
        @test !issorted(permutation)
    end
    checkranges(ranges, data, tree)

    # Run again but don't initialize.
    # Make sure this algorithm is safe to run multiple times.
    empty!(ranges)
    GraphANN.Algorithms.partition!(f, data, tree; executor = single_thread, init = false)
    checkranges(ranges, data, tree)

    # Try again running on multiple threads.
    # Make sure something doesn't go HORRIBLY wrong (just subtly wrong ... )
    empty!(ranges)
    GraphANN.Algorithms.partition!(f, data, tree; executor = dynamic_thread, init = false)
    checkranges(ranges, data, tree)
end

@testset "Testing TPTree" begin
    dims = (1, 4, 5)
    vals = SVector(0.5, 0.2, 0.4)
    x = [1,2,3,4,5]
    y = GraphANN.Algorithms.evalsplit(x, dims, vals)
    @test y == x[1] * vals[1] + x[4] * vals[2] + x[5] * vals[3]

    # Now - perform some tests on some real data.
    data = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, dataset_path)
    _run_tpt_tests(data)

    # Convert element types to UInt8 and try again.
    data = map(x -> map(UInt8, x), data)
    _run_tpt_tests(data)
end

# In this test suite, we see if building trees to group together data points is actuall
# better than just randomly grouping together data points.
#
# The basic idea here is to cluster using the TPTrees, then compute the distances to the
# nearest ~100ish neighbors within each leaf.
#
# Then, we repeat by grouping together consecutive regions of the original data.
# HOPEFULLY, the tree-based clustering does a better job.
@testset "Check that tree clustering actually does something" begin
    data = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, dataset_path)
    numsamples = 1000
    leafsize = 500
    tree = GraphANN.Algorithms.TPTree{4, UInt32}(
        length(data);
        leafsize = leafsize,
        numtrials = 100,
        numsamples = numsamples,
    )

    single_thread = GraphANN.single_thread
    num_neighbors = 100

    # Just group together every 500 data points.
    base_distances = Matrix{Float32}(undef, num_neighbors, length(data))
    for range in GraphANN.batched(1:length(data), leafsize)
        gt = Matrix{GraphANN.Neighbor{UInt32,Float32}}(undef, num_neighbors, length(range))
        dataview = view(data, range)
        GraphANN.bruteforce_search!(gt, dataview, dataview; meter = nothing)
        base_view = view(base_distances, :, range)
        base_view .= GraphANN.getdistance.(gt)
    end

    num_tests = 5
    ranges = Vector{UnitRange{UInt32}}()
    f = x -> push!(ranges, x)

    for i in 1:num_tests
        empty!(ranges)
        GraphANN.Algorithms.partition!(f, data, tree; executor = single_thread, init = true)
        clustered_distances = Matrix{Float32}(undef, num_neighbors, length(data))
        for range in ranges
            # Pre-allocate the result matrix for `bruteforce_search` with eltype `Neighbor`.
            # This will ensure that we get the distances.
            gt = Matrix{GraphANN.Neighbor{UInt32,Float32}}(undef, num_neighbors, length(range))
            dataview = GraphANN.Algorithms.viewdata(data, tree, range)
            @test length(dataview) == length(range)
            GraphANN.bruteforce_search!(gt, dataview, dataview; meter = nothing)

            # Note: We need to translate from the indices returned by `bruteforce_search!` to
            # original indices in the dataset.
            # The position-wise corresponding global indices can be found by the `viewperm`
            # function.
            permview = GraphANN.Algorithms.viewperm(tree, range)
            clustered_view = view(clustered_distances, :, permview)
            clustered_view .= GraphANN.getdistance.(gt)
        end

        # Make sure most of the aggregate sums are closer.
        # These values was empiricly determined by looking at the results.
        # It should be pretty consistent yet close enough to the boundary to detect
        # performance regressions.
        #
        # We're essentially looking at the top 1, 5, 10, etc. nearest neighbors for each
        # data point, checking the accuracy improvement of the tree-based clustering over
        # random clustering at each of those points.
        num_nearest = [1, 5, 10, 20, 50, 100]
        threshold = [1.0, 0.8, 0.85, 0.9, 0.92, 0.92]
        for (t, n) in zip(threshold, num_nearest)
            clustered_sums = sum.(eachcol(view(clustered_distances, 1:n, :)))
            base_sums = sum.(eachcol(view(base_distances, 1:n, :)))
            lt_count = count(clustered_sums .<= base_sums)
            @show n lt_count
            @test lt_count >= t * length(data)
        end
    end
end
