Base.@kwdef mutable struct Clustering
    partition_size::Int
    num_centroids::Int
    preselection_iterations::Int = 3
    preselection_oversample::Float64 = 1.5
    lloyds_iterations::Int = 5
end

function computerecall(groundtruth, ids, n::Int, recall_at::Int)
    vgt = view(groundtruth, 1:n, :)
    vids = view(ids, 1:recall_at, :)

    matches = sum(length.(intersect.(eachcol(vgt), eachcol(vids))))
    return matches / (n * size(groundtruth, 2))
end

function cluster(record, dataset::Dataset, clustering::Clustering; saveprefix = nothing)
    data = load(dataset)
    stats_pre = @elapsed precentroids = GraphANN._Quantization.choose_centroids(
        data,
        clustering.partition_size,
        clustering.num_centroids;
        num_iterations = clustering.preselection_iterations,
        oversample = clustering.preselection_oversample,
    )

    stats_refine = @elapsed refinedcentroids = GraphANN._Quantization.refine(
        precentroids,
        data,
        clustering.num_centroids,
    )

    stats_lloyds = @elapsed finalcentroids = GraphANN._Quantization.lloyds(
        refinedcentroids,
        data;
        num_iterations = clustering.lloyds_iterations,
    )

    if saveprefix !== nothing
        savename = "pq_$(clustering.partition_size)x$(clustering.num_centroids).jls"
        savedir = joinpath(SCRATCH, saveprefix)
        !ispath(savedir) && mkpath(savedir)
        savepath = joinpath(savedir, savename)
        serialize(savepath, finalcentroids)
    else
        savename = nothing
        savepath = nothing
    end

    # Compute the groundtruth for this clustering.
    pqtable = GraphANN.PQTable{size(finalcentroids, 2)}(finalcentroids)
    encoder = GraphANN._Quantization.binned(pqtable)
    encode_type = clustering.num_centroids <= 256 ? UInt8 : UInt16
    encode_time = @elapsed data_encoded = GraphANN._Quantization.encode(encode_type, encoder, data)

    queries = load_queries(dataset)
    groundtruth = load_groundtruth(dataset)

    # Use the PQTable as the metric since it knows how to decode the encoded dataset.
    gttime = @elapsed gt_encoded = 1 .+ GraphANN.bruteforce_search(
        queries,
        data_encoded,
        100;
        metric = pqtable
    )

    results = makeresult([
        dict(dataset),
        dict(clustering),
        Dict(
            :clustering => Dict(
                :preselection => dict(stats_pre; excluded = :value),
                :refinement => dict(stats_refine; excluded = :value),
                :lloyds => dict(stats_lloyds; excluded = :value),
            ),
            :encode_time => encode_time,
            :groundtruth_time => gttime,
            :savename => savename,
            :saveprefix => saveprefix,
            :savepath => savepath,
            :num_threads => Threads.nthreads(),

            # Recalls
            :recall_1at1 => computerecall(groundtruth, gt_encoded, 1, 1),
            :recall_1at5 => computerecall(groundtruth, gt_encoded, 1, 5),
            :recall_1at10 => computerecall(groundtruth, gt_encoded, 1, 10),
            :recall_1at20 => computerecall(groundtruth, gt_encoded, 1, 20),
            :recall_1at100 => computerecall(groundtruth, gt_encoded, 1, 100),

            :recall_5at5 => computerecall(groundtruth, gt_encoded, 5, 5),
            :recall_5at10 => computerecall(groundtruth, gt_encoded, 5, 10),
            :recall_5at20 => computerecall(groundtruth, gt_encoded, 5, 20),
            :recall_5at100 => computerecall(groundtruth, gt_encoded, 5, 100),

            :recall_10at10 => computerecall(groundtruth, gt_encoded, 10, 10),
            :recall_10at20 => computerecall(groundtruth, gt_encoded, 10, 20),
            :recall_10at100 => computerecall(groundtruth, gt_encoded, 10, 100),
        )
    ])
    push!(record, results)
    save(record)
end

#####
##### Cluster Based Query
#####

function quantized_query(
    record::Record,
    quantization::Quantization,
    dataset::Dataset,
    graph::Graph;
    target_accuracies = [0.95, 0.98, 0.99],
    num_neighbors = 1,
    num_loops = 5,
    maxwindow = 400,
    distance_type::QuantizationDistanceType = EagerDistance(),
    distance_strategy::QuantizationDistanceStrategy = EncodedData(),
)

    data, queries, groundtruth = loadall(dataset)
    start_index = GraphANN.medioid(data)
    meta = getmeta(quantization, distance_strategy, dataset, graph)

    # The start index now has to live in the encoded space.
    start = GraphANN.StartNode(start_index, _data_encoded(quantization, dataset)[start_index])
    metric = getmetric(quantization, distance_type)

    # At the moment, I don't have a way of distributing the Eager metric ... so we kind
    # of cheat and wrap it in a `ThreadLocal` for finding the appropriate window sizes.
    thread_local_metric = GraphANN.ThreadLocal(metric)

    # Like the normal query path - we use a memoized closure to find the appropriate window
    # size for the requested accuracies.
    closure = function(windowsize::Integer)
        algo = GraphANN.ThreadLocal(GraphANN.DiskANNRunner(windowsize; costtype = Float32))

        # Since refinement is not yet implemented, we need to use a slightly difference
        # version to calculate recall.
        ids = GraphANN.search(
            algo,
            meta,
            start,
            queries;
            num_neighbors = windowsize,
            metric = thread_local_metric,
        )

        return computerecall(groundtruth, ids, num_neighbors, windowsize)
    end

    memoized = memoize(closure)
    windowsizes = map(target_accuracies) do accuracy
        binarysearch(memoized, accuracy, 1, maxwindow)
    end

    # Now - we actually run the queries.
    callbacks = LatencyCallbacks()
    callback_tuple = get_callbacks(callbacks, SingleThread())
    for (target, windowsize) in zip(target_accuracies, windowsizes)
        algo = GraphANN.DiskANNRunner(windowsize; costtype = Float32)
        f = () -> GraphANN.search(
            algo,
            meta,
            start,
            queries;
            num_neighbors = windowsize,
            callbacks = callback_tuple.callbacks,
            metric = metric,
        )

        reset!(callback_tuple, callbacks)
        ids = f()
        recall = computerecall(groundtruth, ids, num_neighbors, windowsize)

        # Precompile then run.
        repeated(f, num_loops)
        reset!(callback_tuple, callbacks)
        stats = @timed repeated(f, num_loops)
        qps = (num_loops * length(queries)) / stats.time
        results = makeresult([
            dict(stats; excluded = :value),
            dict(dataset),
            dict(graph),
            dict(quantization),
            dict(cb_stats(callback_tuple, callbacks)),
            Dict(
                :target_recall => target,
                :num_neighbors => num_neighbors,
                :windowsize => windowsize,
                :distance_type => distance_type,
                :distance_strategy => distance_strategy,
                :recall => recall,
                :maxwindow => maxwindow,
                :start_index => start_index,
                :qps => qps,
                :num_loops => num_loops,
                :num_queries => length(queries),
                :num_threads => Threads.nthreads(),
                :runtime => stats.time,
            ),
        ])

        push!(record, results)
        save(record)
    end
end
