#####
##### GreedySearch
#####

# Use the `GreedySearch` type to hold parameters and intermediate datastructures
# used to control the greedy search.
mutable struct GreedySearch{M, H, T <: AbstractSet}
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
    best::M
    best_unvisited::H
    visited::T
end

# The default struct
function GreedySearch(search_list_size)
    best = BinaryMinMaxHeap{Neighbor}()
    best_unvisited = BinaryMinMaxHeap{Neighbor}()
    visited = RobinSet{UInt32}()
    return GreedySearch(search_list_size, best, best_unvisited, visited)
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
# TODO; check if type inference works properly.
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
    meta_graph::MetaGraph,
    start_node,
    query,
)
    empty!(algo)

    # Destructure argument
    graph = meta_graph.graph
    data = meta_graph.data

    pushcandidate!(algo, Neighbor(start_node, distance(query, data[start_node])))
    while !done(algo)
        p = getid(getcandidate!(algo))
        neighbors = LightGraphs.outneighbors(graph, p)
        ln = length(neighbors)
        for i in eachindex(neighbors)
            # Perform distance query, and try to prefetch the next datapoint.
            @inbounds v = neighbors[i]
            i < ln && prefetch(data, @inbounds neighbors[i+1])
            @inbounds d = distance(query, data[v])

            ## only bother to add if it's better than the worst currently tracked.
            if !isfull(algo) || d < maximum(algo).distance
                pushcandidate!(algo, Neighbor(v, d))
            end
        end

        # prune
        reduce!(algo)
    end
end

# Single Threaded Query
function searchall(
    algo::GreedySearch,
    meta_graph::MetaGraph,
    start_node,
    queries::AbstractVector;
    num_neighbors = 10
)
    num_queries = length(queries)
    dest = Array{eltype(meta_graph.graph),2}(undef, num_neighbors, num_queries)
    for (col, query) in enumerate(queries)
        search(algo, meta_graph, start_node, query)

        # Copy over the results to the destination
        results = destructive_extract!(algo.best)
        dest_view = view(dest, :, col)
        result_view = view(results, 1:num_neighbors)

        dest_view .= getid.(result_view)
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

    num_queries = length(queries)
    dest = Array{eltype(meta.graph),2}(undef, num_neighbors, num_queries)
    dynamic_thread(eachindex(queries), 64) do r
        for col in r
            query = queries[col]
            algo = tls[]
            search(algo, meta, start_node, query)

            # Copy over the results to the destination
            results = destructive_extract!(algo.best)
            dest_view = view(dest, :, col)
            result_view = view(results, 1:num_neighbors)

            dest_view .= getid.(result_view)
        end
    end

    return dest
end
