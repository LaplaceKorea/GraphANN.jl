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
groundtruth(::Sift1M) = "sift1m.ivecs"
groundtruth(::Sift10M) = "sift10m.ivecs"
groundtruth(::Sift100M) = "sift100m.ivecs"
groundtruth(::Sift1B) = "sift1b.ivecs"

name(::Sift1M) = "sift1m"
name(::Sift10M) = "sift10m"
name(::Sift100M) = "sift100m"
name(::Sift1B) = "sift1b"

graphpath(sift::AbstractSift) = joinpath(SCRATCH, "$(name(sift)).index")

function get_dataset(sift::AbstractSift, allocator = default_allocator(sift))
    return Dataset(;
        # path = "/backup/data/sift1B/bigann_base.bvecs",
        # groundtruth = joinpath("/backup/data/sift1B/gnd", groundtruth(sift)),
        # queries = "/backup/data/sift1B/bigann_query.bvecs",
        path = "/data/sift/bigann_base.bvecs",
        groundtruth = joinpath("/data/sift/groundtruth/queries_10k", groundtruth(sift)),
        queries = "/data/sift/queries/queries_10k.bvecs",
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
    parameters = GraphANN.DiskANNIndexParameters(;
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

    parameters = GraphANN.DiskANNIndexParameters(;
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
    prefetchings = [NoPrefetching()]
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
    sets = [Sift100M()]
    threadings = [MultiThread(), SingleThread()]
    prefetchings = [NoPrefetching()]
    neighbor_sets = [1,5,10]

    #sets = [Sift1M(), Sift10M()]
    #threadings = [SingleThread()]
    #prefetchings = [WithPrefetching()]

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
    sets = [Sift1M(), Sift10M(), Sift100M()]
    #sets = [Sift10M(), Sift100M()]
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

#####
##### Quantization Inference
#####

function __test_quantized_query(record)
    set = Sift1M()
    dataset = get_dataset(set)
    graph = get_graph(set)
    quantization = Quantization(;
        path = joinpath(SCRATCH, "sift1m", "pq_4x256.jls"),
        num_centroids = 256,
        num_partitions = 32,

        encoded_data_allocator = GraphANN.stdallocator,
        pqgraph_allocator = GraphANN.stdallocator,
    )

    dtypes = [EagerDistance(), LazyDistance()]
    dstrategies = [EncodedData(), EncodedGraph()]

    iter = Iterators.product(dtypes, dstrategies)
    for tup in iter
        dtype = tup[1]
        dstrategy = tup[2]

        quantized_query(
            record,
            quantization,
            dataset,
            graph;
            target_accuracies = [0.95],
            distance_type = dtype,
            distance_strategy = dstrategy,
        )
    end
end

function test_quantized_query(record)
    #sets = [Sift1M(), Sift10M(), Sift100M()]
    sets = [Sift100M()]

    dtypes = [EagerDistance(), LazyDistance()]
    dstrategies = [EncodedData(), EncodedGraph()]
    iter = Iterators.product(dtypes, dstrategies)
    clustering_sets = [
        #(num_centroids = 256, num_partitions = 16),
        #(num_centroids = 512, num_partitions = 16),
        (num_centroids = 256, num_partitions = 32),
        #(num_centroids = 512, num_partitions = 32),
    ]

    for set in sets, clustering_set in clustering_sets
        dataset = get_dataset(set)
        graph = get_graph(set)
        # TODO: Hack - fixed dataset element size ...
        @unpack num_partitions, num_centroids = clustering_set
        points_per_partition = div(128, num_partitions)
        quantization = Quantization(;
            path = joinpath(
                SCRATCH,
                name(set),
                "pq_$(points_per_partition)x$(num_centroids).jls"
            ),
            num_centroids = num_centroids,
            num_partitions = num_partitions,
            encoded_data_allocator = GraphANN.stdallocator,
            pqgraph_allocator = default_allocator(set),
        )

        for tup in iter
            dtype = tup[1]
            dstrategy = tup[2]

            quantized_query(
                record,
                quantization,
                dataset,
                graph;
                target_accuracies = [0.95, 0.98, 0.99],
                distance_type = dtype,
                distance_strategy = dstrategy,
            )
        end
    end
end
