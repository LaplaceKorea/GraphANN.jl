struct Neighbor
    id::Int
    distance::Float64
end

getid(x::Int) = x
getid(x::Neighbor) = x.id

idequal(a::Neighbor, b::Neighbor) = (getid(a) == getid(b))

function Base.isless(a::Neighbor, b::Neighbor)
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id)
end

#####
##### GreedySearch
#####

# Use the `GreedySearch` type to hold parameters and intermediate datastructures
# used to control the greedy search.
struct GreedySearch{H, T <: AbstractSet}
    search_list_size::Int

    # Pre-allocated buffer for the search list
    #
    # Strategy with the search list.
    # Maintain the invariant that `queue ⊂ best`.
    # `queue` will used to queue up nodes that have not yet been searched.
    # Since it is a queue, we can easily find the minimum element.
    #
    # When popping off neighbors to get the number of elements in `best` under
    # `search_list_size`, we will also need to pop items off `queue` IF there
    # is a match.
    best::H
    queue::H
    visited::T
end

# The default struct
function GreedySearch(search_list_size)
    best = BinaryMinMaxHeap{Neighbor}()
    queue = BinaryMinMaxHeap{Neighbor}()
    visited = RobinSet{Int}()
    return GreedySearch(search_list_size, best, queue, visited)
end

# Prepare for another run.
function Base.empty!(greedy::GreedySearch)
    empty!(greedy.best)
    empty!(greedy.queue)
    empty!(greedy.visited)
end

Base.length(greedy::GreedySearch) = length(greedy.best)

visited!(greedy::GreedySearch, vertex) = push!(greedy.visited, getid(vertex))

# Get the closest non-visited vertex
getcandidate!(greedy::GreedySearch) = popmin!(greedy.queue)

isfull(greedy::GreedySearch) = length(greedy) >= greedy.search_list_size

function pushcandidate!(greedy::GreedySearch, vertex)
    # If this has already been seen, don't do anything.
    in(getid(vertex), greedy.visited) && return nothing
    visited!(greedy, vertex)

    # TODO: Distance check

    # Since we have not yet visited this vertex, we have to add it both to `best` and `queue`,
    push!(greedy.best, vertex)
    push!(greedy.queue, vertex)
    return nothing
end

done(greedy::GreedySearch) = isempty(greedy.queue)

Base.maximum(greedy::GreedySearch) = _unsafe_maximum(greedy.best)

# Bring the size of the best list down to `search_list_size`
# TODO; check if type inference works properly.
# The function `_unsafe_maximum` can return `nothing`, but Julia should be able to
# handle that
function reduce!(greedy::GreedySearch)
    # Keep ahold of the maximum element in the queue
    # Since `queue` is a subset of `best`, we know that this top element lives in `best`.
    # If the element we pull of the top of `best` matches the top of `queue`, then we
    # need to pop `queue` as well and maintain a new best.
    queue_top = _unsafe_maximum(greedy.queue)
    while length(greedy.best) > greedy.search_list_size
        vertex = popmax!(greedy.best)

        if queue_top !== nothing && idequal(vertex, queue_top)
            popmax!(greedy.queue)
            queue_top = _unsafe_maximum(greedy.queue)
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
        for v in LightGraphs.outneighbors(graph, p)
            # TODO: prefetch next vector
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

# TODO: Thread local storage and multi-threading
function searchall(algo, meta_graph::MetaGraph, start_node, queries)
    for query in queries
        search(algo, meta_graph, start_node, query)
    end
end

