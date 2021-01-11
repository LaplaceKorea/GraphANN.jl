# Test routine to test the implementation logic.
function index_test(record)
    makescratch()

    # Just do Sift1M for testing
    dataset = Dataset(;
        path = "/home/stg/bigann_base.bvecs",
        eltype = GraphANN.Euclidean{128,UInt8},
        maxlines = 1_000_000,
        groundtruth = "/home/stg/projects/sift_versions/sift1m_groundtruth.ivecs",
        queries = "/home/stg/projects/sift_versions/queries.bvecs",
    )

    parameters = GraphANN.GraphParameters(;
        alpha = 1.2,
        window_size = 100,
        target_degree = 128,
        prune_threshold_degree = ceil(Int, 1.3 * 128),
        prune_to_degree = 128,
    )

    index_building(
        record,
        dataset,
        parameters;
        savepath = joinpath(SCRATCH, "sift1m_test.index"),
    )

    cleanup(dataset)
    return nothing
end

function full_index(record)
    makescratch()
    # Just do Sift1M for testing
    datasets = [
        Dataset(;
            path = "/home/stg/bigann_base.bvecs",
            eltype = GraphANN.Euclidean{128,UInt8},
            maxlines = 1_000_000,
            groundtruth = "/home/stg/projects/sift_versions/sift1m_groundtruth.ivecs",
            queries = "/home/stg/projects/sift_versions/queries.bvecs",
        ),
        Dataset(;
            path = "/home/stg/bigann_base.bvecs",
            eltype = GraphANN.Euclidean{128,UInt8},
            maxlines = 10_000_000,
            groundtruth = "/home/stg/projects/sift_versions/sift10m_groundtruth.ivecs",
            queries = "/home/stg/projects/sift_versions/queries.bvecs",
        ),
        Dataset(;
            path = "/home/stg/bigann_base.bvecs",
            eltype = GraphANN.Euclidean{128,UInt8},
            maxlines = 100_000_000,
            groundtruth = "/home/stg/projects/sift_versions/sift100m_groundtruth.ivecs",
            queries = "/home/stg/projects/sift_versions/queries.bvecs",
        ),
    ]

    savepaths = [
        joinpath(SCRATCH, "sift1m.index"),
        joinpath(SCRATCH, "sift10m.index"),
        joinpath(SCRATCH, "sift100m.index"),
    ]

    parameters = GraphANN.GraphParameters(;
        alpha = 1.2,
        window_size = 100,
        target_degree = 128,
        prune_threshold_degree = ceil(Int, 1.3 * 128),
        prune_to_degree = 128,
    )

    for index in eachindex(datasets)
        printstyled(
            "Building Index $index of $(length(datasets))\n";
            color = :green,
            bold = true
        )

        dataset = datasets[index]
        savepath = savepaths[index]

        index_building(
            record,
            dataset,
            parameters;
            savepath = joinpath(SCRATCH, "sift1m_test.index"),
        )

        cleanup(dataset)
    end

    return nothing
end
