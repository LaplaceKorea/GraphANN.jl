#####
##### Graph Search Algorithm
#####

struct SPTAGIndex{G,D<:AbstractVector,T<:Tree}
    graph::G
    data::D
    tree::T
end
_Base.idtype(x::SPTAGIndex) = eltype(x.graph)
Base.eltype(x::SPTAGIndex) = eltype(x.data)

# Callbacks for SPTAG telemetry
Base.@kwdef struct SPTAGCallbacks{A,B}
    prequery::A = donothing
    postquery::B = donothing
end

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

struct SPTAGRunner{I,D,V<:AbstractSet}
    tree_queue::DataStructures.BinaryMinHeap{Neighbor{TreeNode{I},D}}
    graph_queue::DataStructures.BinaryMinHeap{Neighbor{I,D}}
    results::KeepSmallest{Neighbor{I,D}}
    visited::V
end

function SPTAGRunner(
    heapsize::Integer; idtype::Type{I} = UInt32, costtype::Type{D} = Int32
) where {I,D}
    tree_queue = DataStructures.BinaryMinHeap{Neighbor{TreeNode{I},D}}()
    graph_queue = DataStructures.BinaryMinHeap{Neighbor{I,D}}()
    results = KeepSmallest{Neighbor{I,D}}(heapsize)
    # NB: If switching to a `RobinSet`, then rework `ifmissing!` since a slow Base fallback
    # will be used and performance will tank.
    visited = Set{I}()
    return SPTAGRunner(tree_queue, graph_queue, results, visited)
end

function Base.empty!(runner::SPTAGRunner)
    # Slight hijack of DataStructures.jl types.
    # Technically, we don't own "valtree" in the `BinaryMinHeap`.
    empty!(runner.tree_queue.valtree)
    empty!(runner.graph_queue.valtree)
    empty!(runner.results)
    return empty!(runner.visited)
end

# Get the number of neighbors to be returned by the Runner.
_Base.getbound(x::MaybeThreadLocal{SPTAGRunner}) = _Base.getbound(getlocal(x).results)

# The `SPTAGRunner` has two different `Neighbor` types in its heap.
# Here, we use the SPTAGRunner dispatch to auto-construct the "correct" `Neighbor`.
@inline function _Base.Neighbor(runner::SPTAGRunner, id::T, distance::D) where {T,D}
    return Neighbor{T,D}(id, distance)
end

@inline isdone(::TreeCtx, runner::SPTAGRunner) = isempty(runner.tree_queue)
@inline isdone(::GraphCtx, runner::SPTAGRunner) = isempty(runner.graph_queue)

@inline isvisited(x::SPTAGRunner, node) = in(getid(node), x.visited)
@inline visited!(x::SPTAGRunner, node) = push!(x.visited, getid(node))

# Don't mark any nodes during tree search as "visited".
@inline function pushcandidate!(::TreeCtx, runner::SPTAGRunner, candidate::Neighbor)
    return push!(runner.tree_queue, candidate)
end

@inline function pushcandidate!(::GraphCtx, runner::SPTAGRunner, candidate::Neighbor)
    return push!(runner.graph_queue, candidate)
end

@inline function maybe_pushcandidate!(
    ctx::TreeCtx, runner::SPTAGRunner, candidate::Neighbor
)
    isvisited(runner, candidate) && return false
    pushcandidate!(ctx, runner, candidate)
    return true
end

@inline function maybe_pushcandidate!(
    ctx::GraphCtx, runner::SPTAGRunner, candidate::Neighbor
)
    initial_length = length(runner.visited)
    ifmissing!(runner.visited, getid(candidate)) do
        pushcandidate!(ctx, runner, candidate)
    end
    return length(runner.visited) != initial_length
end

@inline getcandidate!(::TreeCtx, x::SPTAGRunner) = pop!(x.tree_queue)
@inline getcandidate!(::GraphCtx, x::SPTAGRunner) = pop!(x.graph_queue)
@inline pushresult!(x::SPTAGRunner, candidate::Neighbor) = push!(x.results, candidate)

@inline isfull(x::SPTAGRunner) = _Base.isfull(x.results)

function init!(runner::SPTAGRunner, tree::Tree, data, query; metric = Euclidean())
    empty!(runner)
    @unpack nodes = tree
    for i in _Trees.rootindices(tree)
        @inbounds child = tree[i]
        distance = evaluate(metric, pointer(data, getid(child)), query)
        candidate = Neighbor(runner, child, distance)
        pushcandidate!(treectx, runner, candidate)
    end

    return nothing
end

function _Base.search(
    runner::SPTAGRunner,
    tree::Tree,
    data::AbstractVector,
    query,
    numleaves::Integer;
    metric = Euclidean(),
)
    leaves_seen = 0
    while !isdone(treectx, runner)
        @inbounds neighbor = getcandidate!(treectx, runner)
        node = getnode(neighbor)

        # Unconditionally add to the queue to process for the graph.
        pair = Neighbor(runner, getid(neighbor), getdistance(neighbor))
        if isleaf(node)
            leaves_seen += Int(maybe_pushcandidate!(graphctx, runner, pair))
            leaves_seen >= numleaves && break
        else
            maybe_pushcandidate!(graphctx, runner, pair)
            for child in _Trees.children(tree, node)
                @unpack id = child
                candidate = Neighbor(
                    runner, child, evaluate(metric, pointer(data, id), query)
                )
                maybe_pushcandidate!(treectx, runner, candidate)
            end
        end
    end
    return leaves_seen
end

early_exit(leaves_seen, maxcheck) = (leaves_seen > maxcheck)
function _Base.search(
    runner::SPTAGRunner,
    index::SPTAGIndex,
    query;
    maxcheck = 2000,
    propagation_limit = 128,
    initial_pivots = 50,
    dynamic_pivots = 4,
    metric = Euclidean(),
    early_exit = always_false,
)
    @unpack graph, data, tree = index
    init!(runner, tree, data, query; metric = metric)
    search(runner, tree, data, query, initial_pivots)

    leaves_seen = 0
    no_better_propagation = 0
    while !isdone(graphctx, runner)
        u = getcandidate!(graphctx, runner)
        neighbors = LightGraphs.outneighbors(graph, getid(u))

        # Prefetch before doing some misc work.
        for vertex in neighbors
            @inbounds prefetch(data, vertex)
        end

        if !isfull(runner) || u <= first(runner.results)
            pushresult!(runner, u)
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
            d = evaluate(metric, pointer(data, v), query)
            leaves_seen += Int(maybe_pushcandidate!(
                graphctx, runner, Neighbor(runner, v, d)
            ))
        end

        if getdistance(first(runner.graph_queue)) > getdistance(first(runner.tree_queue))
            leaves_seen += search(runner, tree, data, query, dynamic_pivots)
        end
    end
    return leaves_seen
end

#####
##### Search All
#####

# single thread
function _Base.search(
    runner::SPTAGRunner,
    index::SPTAGIndex,
    queries::AbstractVector{<:AbstractVector};
    callbacks = SPTAGCallbacks(),
    kw...,
)
    num_queries = length(queries)
    num_neighbors = _Base.getbound(runner)

    dest = Array{idtype(index),2}(undef, num_neighbors, num_queries)
    for (col, query) in enumerate(queries)
        callbacks.prequery()
        search(runner, index, query; kw...)

        # Copy over the results to the destination
        results = destructive_extract!(runner.results)
        for i in 1:num_neighbors
            @inbounds dest[i, col] = getid(results[i])
        end
        callbacks.postquery()
    end
    return dest
end

# multi thread
function _Base.search(
    tls::ThreadLocal{<:SPTAGRunner},
    index::SPTAGIndex,
    queries::AbstractVector{<:AbstractVector};
    callbacks = SPTAGCallbacks(),
    kw...,
)
    metric = Euclidean()
    num_queries = length(queries)
    num_neighbors = _Base.getbound(tls)
    dest = Array{idtype(index),2}(undef, num_neighbors, num_queries)

    dynamic_thread(getpool(tls), eachindex(queries), 64) do col
        callbacks.prequery()
        query = queries[col]
        runner = tls[]

        search(runner, index, query; kw...)

        # Copy over the results to the destination
        results = destructive_extract!(runner.results)
        for i in 1:num_neighbors
            @inbounds dest[i, col] = getid(results[i])
        end
        callbacks.postquery()
    end
    return dest
end
