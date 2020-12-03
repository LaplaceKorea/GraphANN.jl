#####
##### GreedySearch
#####

## Telemetry Types for GreedySearch

# Count number of distance computations
mutable struct DistanceCount
    count::Int
end
DistanceCount() = DistanceCount(0)
reset!(x::DistanceCount) = (x.count = 0)

# Count number of unique vertices
mutable struct VerticesSeen
    count::Int
end
VerticesSeen() = VerticesSeen(0)
reset!(x::VerticesSeen) = (x.count = 0)

mutable struct Latencies
    start::Int
    latencies::Vector{Int}
end
Latencies() = Latencies(0, Int[])
reset!(x::Latencies) = empty!(x.latencies)

# Use the `GreedySearch` type to hold parameters and intermediate datastructures
# used to control the greedy search.
mutable struct GreedySearch{T <: AbstractSet, U <: Telemetry}
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
    telemetry::U
end

# Get the telemetry
telemetry(x::GreedySearch) = x.telemetry

# TODO: Make @generated to ensure constant propagation
reset!(x::Telemetry) = reset!.(Tuple(x.val))

# The default struct
function GreedySearch(search_list_size; kw...)
    best = BinaryMinMaxHeap{Neighbor}()
    best_unvisited = BinaryMinMaxHeap{Neighbor}()
    visited = RobinSet{UInt32}()
    return GreedySearch(
        search_list_size,
        best,
        best_unvisited,
        visited,
        Telemetry(; kw...)
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

        # -- optional telemetry
        ifhasa(telemetry(algo), DistanceCount) do x
            x.count += length(neighbors)
        end
    end

    # -- optional telemetry
    ifhasa(telemetry(algo), VerticesSeen) do x
        x.count += length(algo.visited)
    end

    return nothing
end

# Single Threaded Query
function searchall(
    algo::GreedySearch,
    meta_graph::MetaGraph,
    start_node,
    queries::AbstractVector;
    num_neighbors = 10
)
    reset!(telemetry(algo))
    num_queries = length(queries)
    dest = Array{eltype(meta_graph.graph),2}(undef, num_neighbors, num_queries)
    for (col, query) in enumerate(queries)
        # -- optional telemetry
        ifhasa(telemetry(algo), Latencies) do x
            x.start = time_ns()
        end

        search(algo, meta_graph, start_node, query)

        # Copy over the results to the destination
        results = destructive_extract!(algo.best)
        for i in 1:num_neighbors
            @inbounds dest[i,col] = getid(results[i])
        end

        # -- optional telemetry
        # How long did the round trip take?
        ifhasa(telemetry(algo), Latencies) do x
            push!(x.latencies, time_ns() - x.start)
        end
    end
    return dest
end

# Multi Threaded Query
function searchall(
    tls::ThreadLocal{<:GreedySearch},
    meta::MetaGraph,
    start_node,
    queries::AbstractVector;
    num_neighbors = 10
)
    for algo in getall(tls)
        reset!(telemetry(algo))
    end

    num_queries = length(queries)
    dest = Array{eltype(meta.graph),2}(undef, num_neighbors, num_queries)

    dynamic_thread(eachindex(queries), 64) do r
        for col in r
            query = queries[col]
            algo = tls[]

            # -- optional telemetry
            ifhasa(telemetry(algo), Latencies) do x
                x.start = time_ns()
            end

            search(algo, meta, start_node, query)

            # Copy over the results to the destination
            results = destructive_extract!(algo.best)
            for i in 1:num_neighbors
                @inbounds dest[i,col] = getid(results[i])
            end

            # -- optional telemetry
            # How long did the round trip take?
            ifhasa(telemetry(algo), Latencies) do x
                push!(x.latencies, time_ns() - x.start)
            end
        end
    end

    return dest
end

