#####
##### GreedySearch
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

# Use the `GreedySearch` type to hold parameters and intermediate datastructures
# used to control the greedy search.
mutable struct GreedySearch{T <: AbstractSet}
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
    best::BinaryMinMaxHeap{Neighbor}
    best_unvisited::BinaryMinMaxHeap{Neighbor}
    visited::T
end

function GreedySearch(search_list_size)
    best = BinaryMinMaxHeap{Neighbor}()
    best_unvisited = BinaryMinMaxHeap{Neighbor}()
    visited = RobinSet{UInt32}()
    return GreedySearch(
        search_list_size,
        best,
        best_unvisited,
        visited,
    )
end

# Prepare for another run.
function Base.empty!(greedy::GreedySearch)
    empty!(greedy.best)
    empty!(greedy.best_unvisited)
    empty!(greedy.visited)
end

Base.length(greedy::GreedySearch) = length(greedy.best)

visited!(greedy::GreedySearch, vertex) = push!(greedy.visited, getid(vertex))
getvisited(greedy::GreedySearch) = greedy.visited

# Get the closest non-visited vertex
unsafe_peek(greedy::GreedySearch) = @inbounds greedy.best_unvisited.valtree[1]
getcandidate!(greedy::GreedySearch) = popmin!(greedy.best_unvisited)

isfull(greedy::GreedySearch) = length(greedy) >= greedy.search_list_size

function pushcandidate!(greedy::GreedySearch, vertex)
    # If this has already been seen, don't do anything.
    in(getid(vertex), greedy.visited) && return nothing
    visited!(greedy, vertex)
    # TODO: Distance check?

    # Since we have not yet visited this vertex, we have to add it both to `best` and
    # `best_unvisited`,
    push!(greedy.best, vertex)
    push!(greedy.best_unvisited, vertex)
    return nothing
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

function search(
    algo::GreedySearch,
    meta::MetaGraph,
    start_node,
    query,
    callbacks = GreedyCallbacks(),
)
    empty!(algo)

    # Destructure argument
    @unpack graph, data = meta
    pushcandidate!(algo, Neighbor(start_node, distance(query, data[start_node])))
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

        # Distance computations
        for i in eachindex(neighbors)
            # Perform distance query, and try to prefetch the next datapoint.
            # NOTE: Checking if a vertex has been visited here is actually SLOWER than
            # deferring until after the distance comparison.
            @inbounds v = neighbors[i]
            @inbounds d = distance(query, data[v])

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
function searchall(
    algo::GreedySearch,
    meta_graph::MetaGraph,
    start_node,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = GreedyCallbacks(),
)
    num_queries = length(queries)
    dest = Array{eltype(meta_graph.graph),2}(undef, num_neighbors, num_queries)
    for (col, query) in enumerate(queries)
        # -- optional telemetry
        callbacks.prequery()

        search(algo, meta_graph, start_node, query, callbacks)

        # Copy over the results to the destination
        results = destructive_extract!(algo.best)
        for i in 1:num_neighbors
            @inbounds dest[i,col] = getid(results[i])
        end

        # -- optional telemetry
        callbacks.postquery()
    end
    return dest
end

# Multi Threaded Query
function searchall(
    tls::ThreadLocal{<:GreedySearch},
    meta::MetaGraph,
    start_node,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = GreedyCallbacks(),
)
    num_queries = length(queries)
    dest = Array{eltype(meta.graph),2}(undef, num_neighbors, num_queries)

    dynamic_thread(eachindex(queries), getpool(tls), 64) do r
        for col in r
            query = queries[col]
            algo = tls[]

            # -- optional telemetry
            callbacks.prequery()

            search(algo, meta, start_node, query, callbacks)

            # Copy over the results to the destination
            results = destructive_extract!(algo.best)
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
    return times, GreedyCallbacks(; prequery, postquery)
end

function latency_mt_callbacks()
    times = ThreadLocal(UInt64[])
    prequery = () -> push!(times[], time_ns())
    postquery = () -> begin
        _times = times[]
        _times[end] = time_ns() - _times[end]
    end
    return times, GreedyCallbacks(; prequery, postquery)
end

