function _run_tpt_tests(data::AbstractVector{SVector{N,T}}) where {N,T}
    numsamples = 1000
    leafsize = 500
    numtrials = 1000
    permutation = UInt32.(eachindex(data))
    runner = GraphANN.Algorithms.TPTreeRunner{4}(permutation)

    @test GraphANN.Algorithms.num_split_dimensions(runner) == 4
    @inferred GraphANN.Algorithms.getdims!(data, runner, 1:length(data), numsamples)
    dims = GraphANN.Algorithms.getdims!(data, runner, 1:length(data), numsamples)
    @test isa(dims, NTuple{4,Int})

    # Make sure that the maximum variances are being computed correctly.
    samples = 1:numsamples
    dataview = view(data, samples)
    @test length(dataview) == numsamples
    variances = Statistics.var.(eachrow(reinterpret(reshape, T, dataview)))

    # Get the indices of the ranked variances from highest to lowest
    perm = partialsortperm(variances, 1:4; rev = true)
    @test all(dims .== perm)

    # Next, get normalized weights to maximize the varianze
    @inferred GraphANN.Algorithms.getweights!(data, runner, dims, 1:length(data), numsamples, numtrials)
    weights, meanval = GraphANN.Algorithms.getweights!(data, runner, dims, 1:length(data),numsamples, numtrials)

    # Make sure the return vector is normalized
    @test isapprox(sum(abs2, weights), 1)
    @test isa(weights, SVector{4, Float32})

    # Next - try to sort the whole data.
    # Ensure that all leaves are smaller than the requested leaf size and that the
    # whole data range is covered.
    ranges = GraphANN.Algorithms.partition!(
        data,
        permutation,
        Val(4);
        leafsize = leafsize,
        numtrials = numtrials,
        numsamples = numsamples,
        # Set this low enough so we exercise both the code paths.
        single_thread_threshold = div(length(data), 10),
        init = true
    )

    function checkranges(ranges, data, runner)
        @test !isempty(ranges)
        @test sum(length, ranges) == length(data)
        @test all(x -> length(x) <= leafsize, ranges)
        @test allunique(ranges)
        @test sort(mapreduce(collect, vcat, ranges)) == 1:length(data)

        # Check the `permutation` field. Make sure it is, in fact, still a permutation.
        permutation = runner.permutation
        @test length(permutation) == length(data)
        @test allunique(permutation)
        @test all(x -> in(x, 1:length(data)), permutation)
        @test !issorted(permutation)
    end
    checkranges(ranges, data, runner)

    # Run again but don't initialize.
    # Make sure this algorithm is safe to run multiple times.
    empty!(ranges)
    ranges = GraphANN.Algorithms.partition!(
        data,
        permutation,
        Val(4);
        leafsize = leafsize,
        numtrials = numtrials,
        numsamples = numsamples,
        # Set this low enough so we exercise both the code paths.
        single_thread_threshold = div(length(data), 10),
        init = false
    )
    checkranges(ranges, data, runner)
end

@testset "Testing TPTreeRunner" begin
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
    numtrials = 100
    num_neighbors = 100
    permutation = UInt32.(eachindex(data))

    # Just group together every 500 data points.
    base_distances = Matrix{Float32}(undef, num_neighbors, length(data))
    for range in GraphANN.batched(1:length(data), leafsize)
        erunner = GraphANN.Algorithms.ExhaustiveRunner(
            GraphANN.Neighbor{UInt32,Float32},
            length(range),
            num_neighbors,
        )

        dataview = view(data, range)
        gt = GraphANN.Algorithms.search!(erunner, dataview, dataview; meter = nothing)
        base_view = view(base_distances, :, range)
        base_view .= GraphANN.getdistance.(gt)
    end

    num_tests = 5

    for i in 1:num_tests
        ranges = GraphANN.Algorithms.partition!(
            data,
            permutation,
            Val(4);
            leafsize = leafsize,
            numtrials = numtrials,
            numsamples = numsamples,
            init = true,
            single_thread_threshold = div(length(data), 10),
        )
        clustered_distances = Matrix{Float32}(undef, num_neighbors, length(data))
        for range in ranges
            # Pre-allocate the result matrix for `bruteforce_search` with eltype `Neighbor`.
            # This will ensure that we get the distances.
            erunner = GraphANN.Algorithms.ExhaustiveRunner(
                GraphANN.Neighbor{UInt32,Float32},
                length(range),
                num_neighbors,
            )
            dataview = GraphANN.Algorithms.doubleview(data, permutation, range)
            @test length(dataview) == length(range)
            gt = GraphANN.Algorithms.search!(erunner, dataview, dataview; meter = nothing)

            # Note: We need to translate from the indices returned by `bruteforce_search!` to
            # original indices in the dataset.
            # The position-wise corresponding global indices can be found by the `viewperm`
            # function.
            permview = view(permutation, range)
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
        threshold = [1.0, 0.8, 0.85, 0.9, 0.92, 0.93]
        for (t, n) in zip(threshold, num_nearest)
            clustered_sums = sum.(eachcol(view(clustered_distances, 1:n, :)))
            base_sums = sum.(eachcol(view(base_distances, 1:n, :)))
            lt_count = count(clustered_sums .<= base_sums)
            @show n lt_count
            @test lt_count >= t * length(data)
        end
    end
end
