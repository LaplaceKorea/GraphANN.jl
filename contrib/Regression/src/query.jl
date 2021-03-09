#####
##### Queries
#####

# There are ... a lot of options for parameters running during queries.
# I'm going to try to parameterize these as types and we'll see how well that works out.
abstract type AbstractQueryParameter end
lower(::T) where {T <: AbstractQueryParameter} = string(T)

abstract type AbstractCallbacks <: AbstractQueryParameter end
struct NoCallbacks <: AbstractCallbacks end
struct LatencyCallbacks <: AbstractCallbacks end

abstract type AbstractThreading <: AbstractQueryParameter end
struct SingleThread <: AbstractThreading end
struct MultiThread <: AbstractThreading end

abstract type AbstractPrefetching <: AbstractQueryParameter end
struct NoPrefetching <: AbstractPrefetching end
struct WithPrefetching <: AbstractPrefetching end

# Dispatch Rules - Callbacks
get_callbacks(::NoCallbacks, ::Any) = (callbacks = GraphANN.GreedyCallbacks(),)
get_callbacks(::LatencyCallbacks, ::SingleThread) = GraphANN.Algorithms.latency_callbacks()
get_callbacks(::LatencyCallbacks, ::MultiThread) = GraphANN.Algorithms.latency_mt_callbacks()

# Unpack the names tuple to reset the latencies Vector or ThreadLocal
reset!(x::NamedTuple, ::NoCallbacks) = nothing
reset!(x::NamedTuple, cb::LatencyCallbacks) = reset!(x.latencies, cb)
reset!(x::AbstractVector, ::LatencyCallbacks) = empty!(x)
reset!(x::GraphANN.ThreadLocal, ::LatencyCallbacks) = empty!.(GraphANN.getall(x))

cb_stats(::NamedTuple, ::NoCallbacks) = NamedTuple()
cb_stats(x::NamedTuple, cb::LatencyCallbacks) = cb_stats(x.latencies, cb)
function cb_stats(x::AbstractVector, ::LatencyCallbacks)
    return (
        mean_latency = mean(x),
        max_latency = maximum(x),
        min_latency = minimum(x),
        nines_latency = nines(x),
    )
end
function cb_stats(x::GraphANN.ThreadLocal, cb::LatencyCallbacks)
    x = reduce(vcat, GraphANN.getall(x))
    return cb_stats(x, cb)
end

# Dispatch Rules - Algorithm Construction
start_prefetching(meta, ::NoPrefetching) = nothing
function start_prefetching(meta, ::WithPrefetching)
    GraphANN._Prefetcher.start(meta)
    sleep(0.01)
end

stop_prefetching(meta, ::NoPrefetching) = nothing
function stop_prefetching(meta, ::WithPrefetching)
    sleep(0.01)
    GraphANN._Prefetcher.stop(meta)
end

const MetaGraph = GraphANN.MetaGraph
function make_algo(windowsize::Integer, ::SingleThread, ::NoPrefetching, meta::MetaGraph)
    return GraphANN.DiskANNRunner(windowsize; costtype = Int32), meta
end

function make_algo(windowsize::Integer, ::MultiThread, ::NoPrefetching, meta::MetaGraph)
    return GraphANN.ThreadLocal(GraphANN.DiskANNRunner(windowsize; costtype = Int32)), meta
end

function make_algo(windowsize::Integer, ::SingleThread, ::WithPrefetching, meta::MetaGraph)
    query_pool = GraphANN.ThreadPool(1:1)
    prefetch_pool = GraphANN.ThreadPool(2:2)
    prefetched_meta = GraphANN._Prefetcher.prefetch_wrap(meta, query_pool, prefetch_pool)
    algo = GraphANN.DiskANNRunner(
        windowsize;
        prefetch_queue = GraphANN._Prefetcher.getqueue(prefetched_meta),
        costtype = Int32,
    )

    return algo, prefetched_meta
end

# -- Query top level
function query(
    record::Record,
    dataset::Dataset,
    graph::Graph;
    target_accuracies = [0.95, 0.98, 0.99],
    num_neighbors = 1,
    # Number of times to run queries in a row.
    num_loops = 5,
    # higher level parameters.
    callbacks::AbstractCallbacks = NoCallbacks(),
    threading::AbstractThreading = SingleThread(),
    prefetching::AbstractPrefetching = NoPrefetching(),
    maxwindow = 400,
)
    data, queries, groundtruth = loadall(dataset)
    start_index = GraphANN.medioid(data)
    start = GraphANN.StartNode(start_index, data[start_index])
    meta = GraphANN.MetaGraph(load(graph), data)

    # To find the correct window sizes for the provided accuracies, we create a closure
    # to run the query algorithm and memoize it to make things a little faster.
    closure = function(windowsize::Integer)
        # Even though we might ask for single threaded latency, here we use multithreading
        # just to make convergencde a little faster.
        algo, meta = make_algo(windowsize, MultiThread(), NoPrefetching(), meta)
        ids = GraphANN.searchall(algo, meta, start, queries; num_neighbors = num_neighbors)
        return mean(GraphANN.recall(groundtruth, ids))
    end
    memoized = memoize(closure)

    windowsizes = map(target_accuracies) do accuracy
        binarysearch(memoized, accuracy, 1, maxwindow)
    end

    # Now, unpack the callbacks
    callback_tuple = get_callbacks(callbacks, threading)

    for (target, windowsize) in zip(target_accuracies, windowsizes)
        # Note - the MetaGraph may NOT be modified if prefetching is disabled.
        # However, if prefetching IS enabled, then the MetaGraph will be modified.
        algo, modified_meta = make_algo(windowsize, threading, prefetching, meta)

        # Warmup Round
        f = () -> GraphANN.searchall(
            algo,
            modified_meta,
            start,
            queries;
            num_neighbors = num_neighbors,
            callbacks = callback_tuple.callbacks,
        )

        reset!(callback_tuple, callbacks)
        start_prefetching(modified_meta, prefetching)
        ids = f()
        stop_prefetching(modified_meta, prefetching)

        recall = mean(GraphANN.recall(groundtruth, ids))

        # The query run!
        # Run the `repeated` function once to force precompilaion.
        start_prefetching(modified_meta, prefetching)
        repeated(f, num_loops)
        stop_prefetching(modified_meta, prefetching)
        reset!(callback_tuple, callbacks)

        start_prefetching(modified_meta, prefetching)
        stats = @timed repeated(f, num_loops)
        stop_prefetching(modified_meta, prefetching)

        qps = (num_loops * length(queries)) / stats.time

        # Make an entry in the record
        results = makeresult([
            dict(stats; excluded = :value),
            dict(dataset),
            dict(graph),
            dict(cb_stats(callback_tuple, callbacks)),
            Dict(
                :target_recall => target,
                :num_neighbors => num_neighbors,
                :windowsize => windowsize,
                :recall => recall,
                :callbacks => callbacks,
                :threading => threading,
                :prefetching => prefetching,
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

