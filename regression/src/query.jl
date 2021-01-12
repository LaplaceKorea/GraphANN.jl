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
get_callbacks(::LatencyCallbacks, ::SingleThread) = GraphANN.latency_callbacks()
get_callbacks(::LatencyCallbacks, ::MultiThread) = GraphANN.latency_mt_callbacks()

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
function make_algo(windowsize::Integer, ::SingleThread, ::NoPrefetching, x...)
    return GraphANN.GreedySearch(windowsize)
end

function make_algo(windowsize::Integer, ::MultiThread, ::NoPrefetching, x...)
    return GraphANN.ThreadLocal(GraphANN.GreedySearch(windowsize))
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
        algo = make_algo(windowsize, MultiThread(), NoPrefetching())
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
        algo = make_algo(windowsize, threading, prefetching)

        # Warmup Round
        f = () -> GraphANN.searchall(
            algo,
            meta,
            start,
            queries;
            num_neighbors = num_neighbors,
            callbacks = callback_tuple.callbacks,
        )
        reset!(callback_tuple, callbacks)

        ids = f()
        recall = mean(GraphANN.recall(groundtruth, ids))

        # The query run!
        # Run the `repeated` function once to force precompilaion.
        repeated(f, num_loops)
        reset!(callback_tuple, callbacks)
        stats = @timed repeated(f, num_loops)
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

