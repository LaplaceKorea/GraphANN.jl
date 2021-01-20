#####
##### Graph Search
#####

"""
    GreedyCallbacks

Keyword defined struct - called at different points during query execution.
Allows for arbitrary telemetry to be defined.

Fields and Signatures
---------------------

* `prequery` - Called before a single query is executed. Must take no arguments.
    Does not automatically differentiate between multithreaded and singlethreaded cases.

* `postquery` - Called after a single query is executed. Must take no arguments.
    Does not automatically differentiate between multithreaded and singlethreaded cases.

* `postdistance` - Called after distance computations for a vertex have been completed.
    Signature: `postdistance(algo::GreedySearch, neighbors::AbstractVector)`.
    - `algo` provides the current state of the search.
    - `neighbors` the adjacency list for the vertex that was just processed.
        *NOTE*: Do NOT mutate neighbors, it MUST be constant.
"""
Base.@kwdef struct GreedyCallbacks{A, B, C}
    prequery::A = donothing
    postquery::B = donothing
    postdistance::C = donothing
end

struct StartNode{U,T}
    index::U
    value::T
end

struct HasPrefetching end
struct NoPrefetching end

# Use the `GreedySearch` type to hold parameters and intermediate datastructures
# used to control the greedy search.
mutable struct GreedySearch{T <: AbstractSet, P, D}
    search_list_size::Int

    # Pre-allocated buffer for the search list
    #
    # Strategy with the search list.
    # Maintain the invariant that `best_unvisited ⊂ best`.
    # `best_unvisited` will used to queue up nodes that have not yet been searched.
    # Since it is a queue, we can easily find the minimum element.
    #
    # When popping off neighbors to get the number of elements in `best` under
    # `search_list_size`, we will also need to pop items off `queue` IF there
    # is a match.
    best::BinaryMinMaxHeap{Neighbor{D}}
    best_unvisited::BinaryMinMaxHeap{Neighbor{D}}
    visited::T
    prefetch_queue::P
end

function GreedySearch(
        search_list_size;
        prefetch_queue = nothing,
        cost_type::Type{D} = Float32
   ) where {D}
    best = BinaryMinMaxHeap{Neighbor{D}}()
    best_unvisited = BinaryMinMaxHeap{Neighbor{D}}()
    visited = RobinSet{UInt32}()
    return GreedySearch(
        search_list_size,
        best,
        best_unvisited,
        visited,
        prefetch_queue,
    )
end

# Base prefetching on the existence of a non-nothing element in the prefetch queue
hasprefetching(::GreedySearch{<:AbstractSet, Nothing}) = NoPrefetching()
hasprefetching(::GreedySearch) = HasPrefetching()

# Prepare for another run.
function Base.empty!(greedy::GreedySearch)
    empty!(greedy.best)
    empty!(greedy.best_unvisited)
    empty!(greedy.visited)
end

Base.length(greedy::GreedySearch) = length(greedy.best)

visited!(greedy::GreedySearch, vertex) = push!(greedy.visited, getid(vertex))
isvisited(greedy::GreedySearch, vertex) = in(getid(vertex), greedy.visited)
getvisited(greedy::GreedySearch) = greedy.visited

# Get the closest non-visited vertex
unsafe_peek(greedy::GreedySearch) = @inbounds greedy.best_unvisited.valtree[1]
#getcandidate!(greedy::GreedySearch) = popmin!(greedy.best_unvisited)

getcandidate!(greedy::GreedySearch) = getcandidate!(greedy, hasprefetching(greedy))
getcandidate!(greedy::GreedySearch, ::NoPrefetching) = popmin!(greedy.best_unvisited)

function getcandidate!(greedy::GreedySearch, ::HasPrefetching)
    @unpack best_unvisited, prefetch_queue = greedy
    candidate = popmin!(best_unvisited)

    # If there's another candidate in the queue, try to prefetch it.
    if !isempty(best_unvisited)
        push!(prefetch_queue, getid(unsafe_peek(greedy)))
        _Prefetcher.commit!(prefetch_queue)
    end

    return candidate
end

isfull(greedy::GreedySearch) = length(greedy) >= greedy.search_list_size

function maybe_pushcandidate!(greedy::GreedySearch, vertex)
    # If this has already been seen, don't do anything.
    in(getid(vertex), greedy.visited) && return nothing
    pushcandidate!(greedy, vertex)
end

function pushcandidate!(greedy::GreedySearch, vertex)
    visited!(greedy, vertex)
    @unpack best, best_unvisited = greedy

    # Since we have not yet visited this vertex, we have to add it both to `best` and
    # `best_unvisited`,
    push!(best, vertex)
    push!(best_unvisited, vertex)

    # This vertex is a candidate for prefetching if we pushed it right to the front of
    # the queue.
    if getid(unsafe_peek(greedy)) == getid(vertex)
        maybe_prefetch(greedy, vertex)
    end

    return nothing
end

maybe_prefetch(greedy::GreedySearch, vertex) = maybe_prefetch(greedy, hasprefetching(greedy), vertex)
maybe_prefetch(greedy::GreedySearch, ::NoPrefetching, vertex) = nothing
function maybe_prefetch(greedy::GreedySearch, ::HasPrefetching, vertex)
    @unpack prefetch_queue = greedy
    push!(prefetch_queue, getid(vertex))
    _Prefetcher.commit!(prefetch_queue)
end

done(greedy::GreedySearch) = isempty(greedy.best_unvisited)

Base.maximum(greedy::GreedySearch) = _unsafe_maximum(greedy.best)

# Bring the size of the best list down to `search_list_size`
# TODO: check if type inference works properly.
# The function `_unsafe_maximum` can return `nothing`, but Julia should be able to
# handle that
function reduce!(greedy::GreedySearch)
    # Keep ahold of the maximum element in the best_unvisited
    # Since `best_unvisited` is a subset of `best`, we know that this top element lives
    # in `best`.
    # If the element we pull of the top of `best` matches the top of `best_unvisited`, then we
    # need to pop `queue` as well and maintain a new best.
    top = _unsafe_maximum(greedy.best_unvisited)
    while length(greedy.best) > greedy.search_list_size
        vertex = popmax!(greedy.best)

        if top !== nothing && idequal(vertex, top)
            popmax!(greedy.best_unvisited)
            top = _unsafe_maximum(greedy.best_unvisited)
        end
    end
    return nothing
end

#####
##### Greedy Search Implementation
#####

# Standard distance computation
function _Base.search(
    algo::GreedySearch,
    meta,
    start::StartNode,
    query;
    callbacks = GreedyCallbacks(),
    metric = distance,
)
    empty!(algo)

    # Destructure argument
    @unpack graph, data = meta
    pushcandidate!(algo, Neighbor(start.index, metric(query, start.value)))
    while !done(algo)
        p = getid(unsafe_peek(algo))
        neighbors = LightGraphs.outneighbors(graph, p)

        # Prefetch all new datapoints.
        # IMPORTANT: This is critical for performance!
        for vertex in neighbors
            @inbounds prefetch(data, vertex)
        end

        # Prune
        # Do this here to allow the prefetched vectors time to arrive in the cache.
        getcandidate!(algo)
        reduce!(algo)
        algmax = getdistance(maximum(algo))

        # Distance computations
        for i in eachindex(neighbors)
            # Perform distance query, and try to prefetch the next datapoint.
            # NOTE: Checking if a vertex has been visited here is actually SLOWER than
            # deferring until after the distance comparison.
            @inbounds v = neighbors[i]
            @inbounds d = metric(query, data[v])

            ## only bother to add if it's better than the worst currently tracked.
            if d < algmax || !isfull(algo)
                maybe_pushcandidate!(algo, Neighbor(v, d))
            end
        end

        callbacks.postdistance(algo, neighbors)
    end

    return nothing
end

# Specialize for using PQ based distance computations.
function _Base.search(
    algo::GreedySearch,
    meta::MetaGraph{<:Any, <:PQGraph},
    start::StartNode,
    query;
    callbacks = GreedyCallbacks(),
    metric = distance,
)
    empty!(algo)

    # Destructure argument
    @unpack graph, data = meta
    pushcandidate!(algo, Neighbor(start.index, metric(query, start.value)))
    while !done(algo)
        p = getid(unsafe_peek(algo))
        neighbors = LightGraphs.outneighbors(graph, p)

        # TODO: Implement prefetching for PQGraphs
        # Prefetch all new datapoints.
        # IMPORTANT: This is critical for performance!
        neighbor_points = data[p]
        unsafe_prefetch(neighbor_points, 1, length(neighbor_points))

        # Prune
        # Do this here to allow the prefetched vectors time to arrive in the cache.
        getcandidate!(algo)
        reduce!(algo)

        # Distance computations
        for i in eachindex(neighbors)
            # Since PQ based distance computations take longer, check if we even need
            # to perform the distance computation first.
            @inbounds v = neighbors[i]
            isvisited(algo, v) && continue

            @inbounds d = metric(query, neighbor_points[i])

            ## only bother to add if it's better than the worst currently tracked.
            if d < getdistance(maximum(algo)) || !isfull(algo)
                pushcandidate!(algo, Neighbor(v, d))
            end
        end

        callbacks.postdistance(algo, neighbors)
    end

    return nothing
end

# Single Threaded Query
function _Base.searchall(
    algo::GreedySearch,
    meta,
    start::StartNode,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = GreedyCallbacks(),
    metric::F = distance,
) where {F}
    num_queries = length(queries)
    dest = Array{eltype(meta.graph),2}(undef, num_neighbors, num_queries)
    for (col, query) in enumerate(queries)
        # -- optional telemetry
        callbacks.prequery()

        _Base.distance_prehook(metric, query)
        search(algo, meta, start, query; callbacks, metric)

        # Copy over the results to the destination
        results = destructive_extract!(algo.best, num_neighbors)
        for i in 1:num_neighbors
            @inbounds dest[i,col] = getid(results[i])
        end

        # -- optional telemetry
        callbacks.postquery()
    end
    return dest
end

# Multi Threaded Query
function _Base.searchall(
    tls::ThreadLocal{<:GreedySearch},
    meta,
    start::StartNode,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = GreedyCallbacks(),
    metric::F = distance,
) where {F}
    num_queries = length(queries)
    dest = Array{eltype(meta.graph),2}(undef, num_neighbors, num_queries)

    dynamic_thread(getpool(tls), eachindex(queries), 64) do r
        _metric = _Base.distribute_distance(metric)
        for col in r
            query = queries[col]
            algo = tls[]

            # -- optional telemetry
            callbacks.prequery()

            _Base.distance_prehook(_metric, query)
            search(algo, meta, start, query; callbacks = callbacks, metric = _metric)

            # Copy over the results to the destination
            results = destructive_extract!(algo.best, num_neighbors)
            for i in 1:num_neighbors
                @inbounds dest[i,col] = getid(results[i])
            end

            # -- optional telemetry
            callbacks.postquery()
        end
    end

    return dest
end

#####
##### Callback Implementations
#####

function latency_callbacks()
    times = UInt64[]
    prequery = () -> push!(times, time_ns())
    postquery = () -> times[end] = time_ns() - times[end]
    return (latencies = times, callbacks = GreedyCallbacks(; prequery, postquery))
end

function latency_mt_callbacks()
    times = ThreadLocal(UInt64[])
    prequery = () -> push!(times[], time_ns())
    postquery = () -> begin
        _times = times[]
        _times[end] = time_ns() - _times[end]
    end
    return (latencies = times, callbacks = GreedyCallbacks(; prequery, postquery))
end

