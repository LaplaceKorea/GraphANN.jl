function run_diskann(
    record::Record, index::GraphANN.DiskANNIndex, queries, groundtruth; multithread = false
)
    executor = multithread ? GraphANN.dynamic_thread : GraphANN.single_thread
    runner = GraphANN.DiskANNRunner(index, 200; executor = executor)
    # Function barrier for performance
    return _run_diskann(record, runner, index, queries, groundtruth)
end

function _run_diskann(
    record::Record, runner, index::GraphANN.DiskANNIndex, queries, groundtruth
)
    # Parameter Space
    # Order to the largest search list is first for more accurate progress meters.
    search_list_sizes = reverse(60:60)
    latencies, callbacks = GraphANN.Algorithms.latency_callbacks(runner)

    # Run once to force precompilation
    resize!(runner, maximum(search_list_sizes))
    ids = GraphANN.search(runner, index, queries; num_neighbors = 20, callbacks = callbacks)

    @showprogress 1 for search_list_size in search_list_sizes
        resize!(runner, search_list_size)
        empty!(latencies)
        rt = @elapsed ids = GraphANN.search(
            runner, index, queries; num_neighbors = 20, callbacks = callbacks
        )

        # Compute recalls
        recall_at_20 = mean(GraphANN.recall(groundtruth, ids))
        recall_at_5 = mean(GraphANN.recall(groundtruth, view(ids, 1:5, :)))
        recall_at_1 = mean(GraphANN.recall(groundtruth, view(ids, 1:1, :)))

        # Get the latencies and record some statistics
        times = sort(get(latencies))

        results = makeresult(
            Dict(
                :recall_at_20 => recall_at_20,
                :recall_at_5 => recall_at_5,
                :recall_at_1 => recall_at_1,
                :runtime => rt,
                :qps => length(queries) / rt,
                :search_list_size => search_list_size,
                :num_queries => length(queries),
                :num_threads => Threads.nthreads(),
                # Latencies
                :mean_latency => mean(times),
                :latency_9 => getnine(times, 0.9),
                :latency_99 => getnine(times, 0.99),
                :latency_999 => getnine(times, 0.999),
                :latency_9999 => getnine(times, 0.9999),
                :latency_1 => getnine(times, 0.1),
                :latency_01 => getnine(times, 0.01),
                :latency_001 => getnine(times, 0.001),
                :latency_0001 => getnine(times, 0.0001),
            ),
        )
        push!(record, results)
        save(record)
    end
end
