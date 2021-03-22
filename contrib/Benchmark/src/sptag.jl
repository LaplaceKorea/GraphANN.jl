function run_sptag(
    record::Record,
    index::GraphANN.SPTAGIndex,
    queries,
    groundtruth;
    multithread = false
)
    costtype = GraphANN.costtype(GraphANN.Euclidean(), eltype(index), eltype(queries))
    runner = GraphANN.SPTAGRunner(5; idtype = GraphANN.idtype(index), costtype = costtype)
    if multithread
        runner = GraphANN.ThreadLocal(runner)
    end

    # Function barrier for performance
    return _run_sptag(record, runner, index, queries, groundtruth)
end

function _run_sptag(record::Record, runner, index::GraphANN.SPTAGIndex, queries, groundtruth)
    # Set up parameter space
    # Reverse the order so the largest element is at the front.
    # This ensures that the time computed by the progress bar will be an upper bound rather
    # than a lower bound.
    maxchecks = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] |> reverse
    propagation_limits = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300] |> reverse
    # maxchecks = [500] |> reverse
    # propagation_limits = [20] |> reverse

    iter = Iterators.product(maxchecks, propagation_limits)
    iter_length = prod(length, (maxchecks, propagation_limits))


    # Catch accidentally changing the number of neighbors returned.
    @assert GraphANN._Base.getbound(runner) == 5

    latencies, callbacks = GraphANN.Algorithms.latency_callbacks(runner)

    # Run once for precompilation.
    ids = GraphANN.search(
        runner,
        index,
        queries;
        maxcheck = maximum(maxchecks),
        propagation_limit = maximum(propagation_limits),
        callbacks = callbacks,
    )

    meter = Progress(iter_length, 1)
    for _tuple in iter
        maxcheck = _tuple[1]
        propagation_limit = _tuple[2]
        empty!(latencies)

        rt = @elapsed ids = GraphANN.search(
            runner,
            index,
            queries;
            maxcheck = maxcheck,
            propagation_limit = propagation_limit,
            callbacks = callbacks,
        )

        # Compute recalls
        recall_at_5 = mean(GraphANN.recall(groundtruth, ids))
        recall_at_1 = mean(GraphANN.recall(groundtruth, view(ids, 1:1, :)))

        # Get the latencies and record some statistics
        times = get(latencies) |> sort

        results = makeresult(
            Dict(
                :recall_at_5 => recall_at_5,
                :recall_at_1 => recall_at_1,
                :runtime => rt,
                :qps => length(queries) / rt,
                :maxcheck => maxcheck,
                :propagation_limit => propagation_limit,
                :num_queries => length(queries),
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
            )
        )
        push!(record, results)
        save(record)
        next!(meter)
    end
end
