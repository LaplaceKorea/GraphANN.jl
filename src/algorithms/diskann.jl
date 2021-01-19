module DiskANN

export GraphParameters, generate_index
export GreedySearch, StartNode

# local dependencies
using .._Base
using .._Graphs
using .._Prefetcher
using .._Quantization

# deps
import LightGraphs
import ProgressMeter
import Setfield
import UnPack: @unpack, @pack!

# Constants
const INDEX_BALANCE_FACTOR = 64

#####
##### Index Building
#####

# Simple container for passing around graph parameters.
Base.@kwdef struct GraphParameters
    alpha::Float64
    window_size::Int

    # Three parameters describing graph construction
    target_degree::Int
    prune_threshold_degree::Int
    prune_to_degree::Int
end

function change_threshold(x::GraphParameters, threshold = x.target_degree)
    return Setfield.@set x.prune_threshold_degree = threshold
end

onealpha(x::GraphParameters) = Setfield.@set x.alpha = 1.0

#####
##### Pruner
#####

# This is an auxiliary data structure that aids in the pruning process.
# See tests for basic functionality.
struct Pruner{T}
    # Items to prune
    items::Vector{T}
    # Boolean vector to determine if an index has been pruned.
    pruned::Vector{Bool}
end

Pruner{T}() where {T} = Pruner(T[], Bool[])

function Base.empty!(x::Pruner)
    empty!(x.items)
    empty!(x.pruned)
end

# Mark as `inline` to avoid an allocation caused by creating closures.
#
# NOTE: It's kind of gross to have the `by` and `filter` arguments as normal arguments
# instead of keyword arguments, but when they are keyword arguments, the function closures
# still allocate.
#
# This MAY get fixed in Julia 1.6, which will const-propagate keyword arguments.
@inline function initialize!(
    by::B,
    filter::F,
    x::Pruner,
    itr;
) where {B, F}
    empty!(x)

    # Add all items from the input iterator to the local item list.
    for i in Iterators.filter(filter, itr)
        push!(x.items, by(i))
    end

    # Intially, mark all items as valid (i.e. not pruned)
    resize!(x.pruned, length(x))
    x.pruned .= false
    return nothing
end
initialize!(x::Pruner, itr) = initialize!(identity, _ -> true, x, itr)

# NB: Only call BEFORE performing any pruning
# This WILL NOT maintain the pruned state correctly.
Base.sort!(x::Pruner, args...; kw...) = sort!(x.items, args...; kw...)

Base.length(x::Pruner) = length(x.items)
Base.eltype(x::Pruner) = eltype(x.items)

# Since intermediate items may be pruned, we need to indicate that the size of this
# type cannot be known in advance.
Base.IteratorSize(::Type{<:Pruner}) = Base.SizeUnknown()

ispruned(x::Pruner, i) = @inbounds(x.pruned[i])
prune!(x::Pruner, i) = @inbounds(x.pruned[i] = true)

function Base.iterate(x::Pruner, i = 1)
    # Move forward until we find a non-removed entry
    while (i <= length(x) && ispruned(x, i))
        i += 1
    end
    return i > length(x) ? nothing : (@inbounds(x.items[i]), (i + 1))
end

# Mark entries as pruned according to some function `F`.
function prune!(f::F, x::Pruner; start = 1) where {F}
    for i in start:length(x)
        @inbounds begin
            if !ispruned(x, i) && f(x.items[i])
                prune!(x, i)
            end
        end
    end
end

function Base.sizehint!(x::Pruner, sz)
    sizehint!(x.items, sz)
    sizehint!(x.pruned, sz)
end

#####
##### Vemana Indexing
#####

# Some implementation details
#
# 1. The algorithms for sorting are manually set to `QuickSort`.
#   This is because Julia defaults to `MergeSort`, which allocates temporary arrays.
#   On its own, this isn't too much of a problem, but allocations quickly become
#   troublesome when dealing with multi-threaded code.
#
#   `QuickSort`, on the other hand, does NOT allocate, so we use it.
#   TODO: `InsertionSort` might be something to look at if our vectors to sort are
#   pretty small ...
#
# 2. We try to make agressive resuse of existing allocations.
#   Types like the `NextListBuffer` below are abstractions to help make this optimization
#   a little easier to deal with in implementation code.

#####
##### NextListBuffer
#####

struct NextListBuffer{T}
    # Keep around previously allocated buffers
    buffers::Vector{Vector{T}}
    nextlists::Dict{Int, Vector{T}}
    sizehint::Int
end

# Constructor
function NextListBuffer{T}(sizehint::Integer, num_buffers::Integer = 1) where {T}
    buffers = map(1:num_buffers) do _
        v = T[]
        sizehint!(v, sizehint)
        return v
    end

    nextlists = Dict{Int, Vector{T}}()
    sizehint!(nextlists, num_buffers)
    return NextListBuffer{T}(buffers, nextlists, sizehint)
end

# If we have a previously allocated vector, return that.
# Otherwise, allocate a new buffer.
function Base.get!(x::NextListBuffer{T}) where {T}
    !isempty(x.buffers) && return pop!(x.buffers)
    new = T[]
    sizehint!(new, x.sizehint)
    return new
end

Base.getindex(x::NextListBuffer, i) = x.nextlists[i]
Base.setindex!(x::NextListBuffer{T}, v::Vector{T}, i) where {T} = x.nextlists[i] = v

Base.pairs(x::NextListBuffer) = pairs(x.nextlists)

# Remove previous mappings but save the allocated vectors for the next iteration.
function Base.empty!(x::NextListBuffer)
    for v in values(x.nextlists)
        empty!(v)
        push!(x.buffers, v)
    end
    empty!(x.nextlists)
end

# Actuall empty everything.
function purge!(x::NextListBuffer)
    empty!(x.buffers)
    empty!(x.nextlists)
end

# Dispatch on behavior based on Sets vs Arrays.
# Use a closure to avoid code that doesn't necessarily need to run.
maybeunion!(f::F, candidates::AbstractSet) where {F} = union!(candidates, f())
maybeunion!(f::F, candidates::AbstractArray) where {F} = nothing

"""
    neighbor_updates!(args...)

This function roughly implements the `RobustPrune` algorithm presented in the paper.
One difference is that instead of directly mutating the graph, we populate a `nextlist`
that will contain the next neighbors for this node.

This allows threads to work in parallel with periodic synchronous updates to the graph.

*Note*: I'm pretty sure this technique is how it's implemented in the C++ code.

* `nextlist` - Vector to be populated for the next neighbors of the current node.
* `candidates` - Candidate nodes to be the next neighbors
* `vertex` - The current vertex this is being applied to.
* `meta` - Combination of Graph and Dataset.
* `parameters` - Parameters governing graph construction.
* `pruner` - Preallocated `Pruner` to help remove neighbors that fall outside the
    distance threshold defined by `parameters.alpha`
"""
function neighbor_updates!(
    nextlist::AbstractVector,
    candidates,
    vertex,
    meta::MetaGraph,
    pruner::Pruner,
    alpha::AbstractFloat,
    target_degree::Integer
)
    # Destructure some parameters
    @unpack graph, data = meta

    # If the passed `candidates` is a Set, then append the current out neighbors
    # of the query vertex to the set.
    #
    # However, if `candidates` is an Array, than don't do anything because that array
    # likely comes from a graph and we do NOT want to mutate it.
    maybeunion!(() -> LightGraphs.outneighbors(graph, vertex), candidates)

    # Use lazy functions to efficientaly initialize the Pruner object
    vertex_data = data[vertex]
    initialize!(
        u -> Neighbor(u, distance(vertex_data, @inbounds data[u])),
        !isequal(vertex),
        pruner,
        candidates,
    )
    sort!(pruner; alg = Base.QuickSort)

    # As a precaution, make sure the list of updates for this node is empty.
    empty!(nextlist)

    # Manually lower the `for` loop to iteration so we can get the internal state of the
    # pruner iterator.
    #
    # This lets us start at the current location in the iterator to reduce the number
    # of comparisons.
    next = iterate(pruner)
    while next !== nothing
        (i, state) = next
        push!(nextlist, getid(i))

        # Note: We're indexing `data` with `Neighbor` objects, but that's fine because
        # we've defined that behavior in `utils.jl`.
        f = x -> (alpha * distance(data[i], data[x]) <= getdistance(x))
        prune!(f, pruner; start = state)
        length(nextlist) >= target_degree && break

        # Next step in the iteration interface
        next = iterate(pruner, state)
    end
end

function apply_nextlists!(graph, locks, tls::ThreadLocal; empty = false)
    # First step - copy over the next lists
    on_threads(allthreads()) do
        # Get local storage for this thread.
        # No need to lock because work partitioning is done such that the roote
        # nodes are unique for each thread.
        #
        # ... however, sneaky bugs can sometimes occur if this invariante,
        # so lock just to be on the safe side.
        #
        # Fortunately, the chance of collision is pretty low, so it shouldn't take very
        # long to get the locks.
        storage = tls[]
        for (u, neighbors) in pairs(storage.nextlists)
            Base.@lock locks[u] copyto!(graph, u, neighbors)
        end

        empty && empty!(storage.nextlists)
    end
end

function add_backedges!(graph, locks, tls, needs_pruning, prune_threshold)
    # Merge all edges to add together
    on_threads(allthreads()) do
        storage = tls[]
        for (u, neighbors) in pairs(storage.nextlists)
            for v in neighbors
                n = Base.@lock locks[v] LightGraphs.add_edge!(graph, v, u)
                n > prune_threshold && (needs_pruning[v] = true)
            end
        end
        empty!(storage.nextlists)
    end
end

function prune!(
    meta::MetaGraph,
    parameters::GraphParameters,
    tls::ThreadLocal,
    needs_pruning::Vector{Bool},
)
    @unpack graph = meta
    @unpack prune_to_degree, alpha = parameters

    dynamic_thread(eachindex(needs_pruning), INDEX_BALANCE_FACTOR) do v
        needs_pruning[v] || return nothing

        storage = tls[]
        nextlist = get!(storage.nextlists)

        neighbor_updates!(
            nextlist,
            # Since we've already added the back edge, the source vertex that caused the
            # overflow is already in the adjacency list for `v`
            LightGraphs.outneighbors(graph, v),
            v,
            meta,
            storage.pruner,
            alpha,
            prune_to_degree,
        )
        storage.nextlists[v] = nextlist
        needs_pruning[v] = false
        return nothing
    end
end

# Commit all the pending updates to the graph.
# Then, cycle util no nodes violate the degree requirement
function commit!(
    meta::MetaGraph,
    parameters::GraphParameters,
    locks::AbstractVector,
    needs_pruning::Vector{Bool},
    tls::ThreadLocal,
)
    @unpack graph = meta
    @unpack prune_threshold_degree = parameters

    # Step 1 - update all the next lists
    apply_nextlists!(graph, locks, tls)

    # Step 2 - Add back edges
    add_backedges!(graph, locks, tls, needs_pruning, prune_threshold_degree)

    # Step 3 - process all over subscribed vertices
    prune!(meta, parameters, tls, needs_pruning)

    # Step 4 - update nextlists for all over subscribed vertices
    apply_nextlists!(graph, locks, tls; empty = true)
    return nothing
end

"""
Generate the Index for a dataset
"""
function generate_index(
    data,
    parameters::GraphParameters;
    batchsize = 1000,
    allocator = stdallocator,
    graph_type = DefaultAdjacencyList{UInt32},
    no_progress = false
)
    @unpack window_size, target_degree, prune_threshold_degree = parameters

    # Generate a random `max_degree` regular graph.
    # TODO: Also, default the graph to UInt32's to save on space.
    # Keep this as a parameter so we can promote to Int64's if needed.
    graph = random_regular(
        graph_type,
        length(data),
        target_degree;
        max_edges = prune_threshold_degree,
        allocator = allocator,
    )

    # Create a spin-lock for each vertex in the graph.
    # This will be used during the `commit!` function to synchronize access to the graph.
    locks = [Base.Threads.SpinLock() for _ in 1:LightGraphs.nv(graph)]
    needs_pruning = [false for _ in 1:LightGraphs.nv(graph)]

    meta = MetaGraph(graph, data)
    tls = ThreadLocal(;
        greedy = GreedySearch(window_size),
        pruner = Pruner{Neighbor}(),
        nextlists = NextListBuffer{eltype(graph)}(
            target_degree,
            2 * ceil(Int, batchsize / Threads.nthreads()),
        ),
    )

    # First iteration - set alpha = 1.0
    # Second iteration - decrease pruning threshold to `target_degree`
    #for i in 1:2
    for i in 1:2
        _parameters = (i == 1) ? onealpha(parameters) : parameters
        _generate_index(
            meta,
            _parameters,
            tls,
            locks,
            needs_pruning,
            batchsize,
            no_progress = no_progress,
        )
    end

    # Final cleanup - enforce degree constraint.
    _parameters = change_threshold(parameters)
    @unpack prune_threshold_degree = _parameters
    Threads.@threads for v in LightGraphs.vertices(graph)
        if length(LightGraphs.outneighbors(graph, v)) > prune_threshold_degree
            @inbounds needs_pruning[v] = true
        end
    end
    prune!(meta, _parameters, tls, needs_pruning)
    apply_nextlists!(graph, locks, tls; empty = true)

    return meta
end

@noinline function _generate_index(
    meta::MetaGraph,
    parameters::GraphParameters,
    tls::ThreadLocal,
    locks::AbstractVector,
    needs_pruning::Vector{Bool},
    batchsize::Integer;
    no_progress = false
)
    @unpack graph, data = meta
    @unpack alpha, target_degree = parameters

    # TODO: Shuffle visit order.
    num_batches = ceil(Int, length(data) / batchsize)
    progress_meter = ProgressMeter.Progress(num_batches, 1, "Computing Index...")
    for r in batched(1:length(data), batchsize)
        # Use dynamic load balancing.
        itertime = @elapsed dynamic_thread(r, INDEX_BALANCE_FACTOR) do vertex
            # Get thread local storage
            storage = tls[]

            # Perform a greedy search from this node.
            # The visited list will live inside the `greedy` object and will be extracted
            # using the `getvisited` function.
            datum = data[vertex]
            search(storage.greedy, meta, StartNode(vertex, datum), datum)
            candidates = getvisited(storage.greedy)

            # Run the `RobustPrune` algorithm on the graph starting at this point.
            # Array `nextlist` will contain the updated neighbors for this vertex.
            # We delay actually implementing the updates to facilitate parallelization.
            nextlist = get!(storage.nextlists)
            neighbor_updates!(
                nextlist,
                candidates,
                vertex,
                meta,
                storage.pruner,
                alpha,
                target_degree,
            )
            storage.nextlists[vertex] = nextlist
        end

        # Update the graph.
        synctime = @elapsed commit!(
            meta,
            parameters,
            locks,
            needs_pruning,
            tls,
        )

        no_progress || ProgressMeter.next!(
            progress_meter;
            showvalues = ((:iter_time, itertime), (:sync_time, synctime)),
        )
    end
    return graph
end

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
mutable struct GreedySearch{T <: AbstractSet, P}
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
    prefetch_queue::P
end

function GreedySearch(search_list_size; prefetch_queue = nothing)
    best = BinaryMinMaxHeap{Neighbor}()
    best_unvisited = BinaryMinMaxHeap{Neighbor}()
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

        # Distance computations
        for i in eachindex(neighbors)
            # Perform distance query, and try to prefetch the next datapoint.
            # NOTE: Checking if a vertex has been visited here is actually SLOWER than
            # deferring until after the distance comparison.
            @inbounds v = neighbors[i]
            @inbounds d = metric(query, data[v])

            ## only bother to add if it's better than the worst currently tracked.
            if d < getdistance(maximum(algo)) || !isfull(algo)
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


end # module
