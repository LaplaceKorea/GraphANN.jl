abstract type  AbstractSift end
struct Sift1M <: AbstractSift end
struct Sift10M <: AbstractSift end
struct Sift100M <: AbstractSift end
struct Sift1B <: AbstractSift end

default_allocator(::Sift1M) = GraphANN.stdallocator
default_allocator(::AbstractSift) = GraphANN.pmallocator("/mnt/public")

maxlines(::Sift1M) = 1_000_000
maxlines(::Sift10M) = 10_000_000
maxlines(::Sift100M) = 100_000_000
maxlines(::Sift1B) = 1_000_000_000

# for airval
# groundtruth(::Sift1M) = "idx_1M.ivecs"
# groundtruth(::Sift10M) = "idx_10M.ivecs"
# groundtruth(::Sift100M) = "idx_100M.ivecs"
# groundtruth(::Sift1B) = "idx_1000M.ivecs"

# for air3
groundtruth(::Sift1M) = "sift1m_groundtruth.ivecs"
groundtruth(::Sift10M) = "sift10m_groundtruth.ivecs"
groundtruth(::Sift100M) = "sift100m_groundtruth.ivecs"
groundtruth(::Sift1B) = "sift1b_groundtruth.ivecs"

name(::Sift1M) = "sift1m"
name(::Sift10M) = "sift10m"
name(::Sift100M) = "sift100m"
name(::Sift1B) = "sift1b"

graphpath(sift::AbstractSift) = joinpath(SCRATCH, "$(name(sift)).index")

function get_dataset(sift::AbstractSift, allocator = default_allocator(sift))
    return Dataset(;
        #path = "/backup/data/sift1B/bigann_base.bvecs",
        #groundtruth = joinpath("/backup/data/sift1B/gnd", groundtruth(sift)),
        #queries = "/backup/data/sift1B/bigann_query.bvecs",
        path = "/home/stg/bigann_base.bvecs",
        groundtruth = joinpath("/home/stg/projects/sift_versions", groundtruth(sift)),
        queries = "/home/stg/projects/sift_versions/queries.bvecs",
        eltype = GraphANN.Euclidean{128,UInt8},
        maxlines = maxlines(sift),
        data_allocator = allocator,
    )
end

function get_graph(sift::AbstractSift, allocator = default_allocator(sift))
    return Graph(; path = graphpath(sift), graph_allocator = allocator)
end

# Test routine to test the implementation logic.
function __test_index(record::Record)
    makescratch()

    # Just do Sift1M for testing
    dataset = get_datset(Sift1M())
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
        get_dataset(Sift1M()),
        get_dataset(Sift10M()),
        get_dataset(Sift100M()),
    ]

    savepaths = [
        "sift1m.index",
        "sift10m.index",
        "sift100m.index",
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
        allocator = dataset.data_allocator

        index_building(
            record,
            dataset,
            parameters;
            savepath = joinpath(SCRATCH, savepath),
            allocator = allocator,
        )

        cleanup(dataset)
    end

    return nothing
end

#####
##### Query
#####

# Query Performance - No Product Quantization
function __test_query(record::Record)
    sift = Sift1M()
    dataset = get_dataset(sift)
    graph = get_graph(sift)

    threadings = [SingleThread(), MultiThread()]
    prefetchings = [NoPrefetching(), WithPrefetching()]
    num_neighbors = [5]

    iter = Iterators.product(num_neighbors, prefetchings, threadings)

    for tup in iter
        num_neighbors = tup[1]
        prefetching = tup[2]
        threading = tup[3]

        if prefetching == WithPrefetching() && threading == MultiThread()
            continue
        end

        Regression.query(
            record,
            dataset,
            graph;
            num_neighbors = num_neighbors,
            callbacks = LatencyCallbacks(),
            threading = threading,
            prefetching = prefetching,
        )
    end
    return nothing
end

function test_query(record::Record)
    sets = [Sift1M(), Sift10M(), Sift100M()]
    threadings = [MultiThread(), SingleThread()]
    prefetchings = [NoPrefetching(), WithPrefetching()]

    #sets = [Sift1M(), Sift10M()]
    #threadings = [SingleThread()]
    #prefetchings = [WithPrefetching()]
    neighbor_sets = [1,5,10]

    iter = Iterators.product(neighbor_sets, prefetchings, threadings)
    for set in sets
        dataset = get_dataset(set)
        graph = get_graph(set)

        for tup in iter
            num_neighbors = tup[1]
            prefetching = tup[2]
            threading = tup[3]

            if prefetching == WithPrefetching() && threading == MultiThread()
                continue
            end

            Regression.query(
                record,
                dataset,
                graph;
                num_neighbors = num_neighbors,
                callbacks = LatencyCallbacks(),
                threading = threading,
                prefetching = prefetching,
            )
        end

        cleanup(dataset)
        cleanup(graph)
    end
end

#####
##### Clustering
#####

function __test_clustering(record::Record)
    dataset = get_dataset(Sift1M())
    clustering = Clustering(;
        partition_size = 4,
        num_centroids = 256,
    )

    cluster(record, dataset, clustering)
    return nothing
end

function test_clustering(record::Record)
    #sets = [Sift1M(), Sift10M(), Sift100M()]
    sets = [Sift10M(), Sift100M()]
    clusterings = [
        Clustering(partition_size = 8, num_centroids = 256),
        Clustering(partition_size = 8, num_centroids = 512),
        Clustering(partition_size = 4, num_centroids = 256),
        Clustering(partition_size = 4, num_centroids = 512),
    ]

    for set in sets
        dataset = get_dataset(set, GraphANN.stdallocator)
        for clustering in clusterings
            cluster(record, dataset, clustering; saveprefix = name(set))
        end
    end
end
