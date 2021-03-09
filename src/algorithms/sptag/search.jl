#####
##### Graph Search Algorithm
#####

# Use Contexts to describe to the SPTAG search algorithm if we're searching through a
# tree or through the neighborhood graph.
#
# This allows us to overload common methods to keep similar teminology between the graph
# search and tree search algorithms.
abstract type AbstractTagCtx end
struct TreeCtx <: AbstractTagCtx end
struct GraphCtx <: AbstractTagCtx end

# Useful Aliases
const treectx = TreeCtx()
const graphctx = GraphCtx()

# TODO: I'm not so sure about these definitions ...
# The idea is for `getid` to ALWAYS return an Integer when calling `getid`.
getnode(x::Neighbor{<:TreeNode}) = x.id
_Base.getid(x::Neighbor{<:TreeNode}) = getid(getnode(x))

struct TagSearch{I, D, V <: AbstractSet}
    tree_queue::DataStructures.BinaryMinHeap{Neighbor{TreeNode{I}, D}}
    graph_queue::DataStructures.BinaryMinHeap{Neighbor{I, D}}
    results::KeepSmallest{Neighbor{I, D}}
    visited::V
end

function TagSearch(
    heapsize::Integer;
    idtype::Type{I} = UInt32,
    costtype::Type{D} = Int32,
) where {I,D}
    tree_queue = DataStructures.BinaryMinHeap{Neighbor{TreeNode{I}, D}}()
    graph_queue = DataStructures.BinaryMinHeap{Neighbor{I, D}}()
    results = KeepSmallest{Neighbor{I,D}}(heapsize)
    visited = RobinSet{I}()
    return TagSearch(tree_queue, graph_queue, results, visited)
end

function Base.empty!(algo::TagSearch)
    # Slight hijack of DataStructures.jl types.
    # Technically, we don't own "valtree" in the `BinaryMinHeap`.
    empty!(algo.tree_queue.valtree)
    empty!(algo.graph_queue.valtree)
    empty!(algo.results)
    empty!(algo.visited)
end

# The `TreeSearcher` has two different `Neighbor` types in its heap.
# Here, we use the TreeSearcher dispatch to auto-construct the "correct" `Neighbor`.
function _Base.Neighbor(algo::TagSearch, id::T, distance::D) where {T,D}
    return Neighbor{T,D}(id, distance)
end

isdone(::TreeCtx, algo::TagSearch) = isempty(algo.tree_queue)
isdone(::GraphCtx, algo::TagSearch) = isempty(algo.graph_queue)

isvisited(x::TagSearch, node) = in(getid(node), x.visited)
visited!(x::TagSearch, node) = push!(x.visited, getid(node))

# Don't mark any nodes during tree search as "visited".
function pushcandidate!(::TreeCtx, algo::TagSearch, candidate::Neighbor)
    return push!(algo.tree_queue, candidate)
end

function pushcandidate!(::GraphCtx, algo::TagSearch, candidate::Neighbor)
    visited!(algo, candidate)
    return push!(algo.graph_queue, candidate)
end

function maybe_pushcandidate!(ctx::AbstractTagCtx, algo::TagSearch, candidate::Neighbor)
    isvisited(algo, candidate) && return false
    pushcandidate!(ctx, algo, candidate)
    return true
end

getcandidate!(::TreeCtx, x::TagSearch) = pop!(x.tree_queue)
getcandidate!(::GraphCtx, x::TagSearch) = pop!(x.graph_queue)
pushresult!(x::TagSearch, candidate::Neighbor) = push!(x.results, candidate)

isfull(x::TagSearch) = _Base.isfull(x.results)

function init!(algo::TagSearch, tree::Tree, data, query; metric = Euclidean())
    empty!(algo)
    @unpack nodes = tree
    for i in _Trees.rootindices(tree)
        child = tree[i]
        candidate = Neighbor(algo, child, evaluate(metric, data[getid(child)], query))
        pushcandidate!(treectx, algo, candidate)
    end

    return nothing
end

function _Base.search(
    algo::TagSearch,
    tree::Tree,
    data::AbstractVector,
    query,
    numleaves::Integer;
    metric = Euclidean(),
)
    leaves_seen = 0
    while !isdone(treectx, algo)
        @inbounds neighbor = getcandidate!(treectx, algo)
        node = getnode(neighbor)

        # Unconditionally add to the queue to process for the graph.
        pair = Neighbor(algo, getid(neighbor), getdistance(neighbor))
        if isleaf(node)
            leaves_seen += Int(maybe_pushcandidate!(graphctx, algo, pair))
            leaves_seen >= numleaves && break
        else
            maybe_pushcandidate!(graphctx, algo, pair)
            for child in _Trees.children(tree, node)
                @unpack id = child
                candidate = Neighbor(algo, child, evaluate(metric, data[id], query))
                maybe_pushcandidate!(treectx, algo, candidate)
            end
        end
    end
    return leaves_seen
end

early_exit(leaves_seen, maxcheck) = (leaves_seen > maxcheck)
function _Base.search(
    algo::TagSearch,
    meta::MetaGraph,
    tree::Tree,
    query;
    maxcheck = 2000,
    propagation_limit = 128,
    initial_pivots = 50,
    dynamic_pivots = 4,
    metric = Euclidean(),
    early_exit = always_false,
)
    @unpack graph, data = meta
    init!(algo, tree, data, query; metric = metric)
    search(algo, tree, data, query, initial_pivots)

    leaves_seen = 0
    no_better_propagation = 0
    while !isdone(graphctx, algo)
        u = getcandidate!(graphctx, algo)
        neighbors = LightGraphs.outneighbors(graph, getid(u))

        # Prefetch before doing some misc work.
        for vertex in neighbors
            @inbounds prefetch(data, vertex)
        end

        if !isfull(algo) || u <= first(algo.results)
            pushresult!(algo, u)
            no_better_propagation = 0
        else
            no_better_propagation += 1
            if no_better_propagation > propagation_limit || leaves_seen > maxcheck
                break
            end
        end

        # Maybe check the number of leaves seen here and abort early.
        early_exit(leaves_seen, maxcheck) && break

        # Expand neighborhood
        for v in LightGraphs.outneighbors(graph, getid(u))
            d = evaluate(metric, data[v], query)
            leaves_seen += Int(maybe_pushcandidate!(graphctx, algo, Neighbor(algo, v, d)))
        end

        if getdistance(first(algo.graph_queue)) > getdistance(first(algo.tree_queue))
            leaves_seen += search(algo, tree, data, query, dynamic_pivots)
        end
    end
    return leaves_seen
end

#####
##### Search All
#####

function _Base.searchall(
    algo::TagSearch,
    meta,
    tree,
    queries::AbstractVector;
    kw...
)
    @unpack graph = meta
    num_queries = length(queries)
    # TODO: This is so gross ...
    num_neighbors = _Base.getbound(algo.results)

    dest = Array{eltype(graph),2}(undef, num_neighbors, num_queries)
    for (col, query) in enumerate(queries)
        search(algo, meta, tree, query; kw...)

        # Copy over the results to the destination
        results = destructive_extract!(algo.results)
        for i in 1:num_neighbors
            @inbounds dest[i,col] = getid(results[i])
        end
    end
    return dest
end

function _Base.searchall(
    tls::ThreadLocal{<:TagSearch},
    meta,
    tree,
    queries;
    kw...
)
    metric = Euclidean()
    num_queries = length(queries)
    # TODO: This is so gross ...
    num_neighbors = _Base.getbound(tls[1].results)
    dest = Array{eltype(meta.graph), 2}(undef, num_neighbors, num_queries)

    dynamic_thread(getpool(tls), eachindex(queries), 64) do col
        query = queries[col]
        algo = tls[]

        search(algo, meta, tree, query; kw...)

        # Copy over the results to the destination
        results = destructive_extract!(algo.results)
        for i in 1:num_neighbors
            @inbounds dest[i, col] = getid(results[i])
        end
    end
    return dest
end

