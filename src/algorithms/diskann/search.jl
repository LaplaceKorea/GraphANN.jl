#####
##### Graph Search
#####

"""
    DiskANNCallbacks

Keyword defined struct - called at different points during query execution.
Allows for arbitrary telemetry to be defined.

Fields and Signatures
---------------------

* `prequery` - Called before a single query is executed. Must take no arguments.
    Does not automatically differentiate between multithreaded and singlethreaded cases.

* `postquery` - Called after a single query is executed. Must take no arguments.
    Does not automatically differentiate between multithreaded and singlethreaded cases.

* `postdistance` - Called after distance computations for a vertex have been completed.
    Signature: `postdistance(algo::DiskANNRunner, vertex::Integer, neighbors::AbstractVector)`.
    - `algo` provides the current state of the search.
    - `neighbors` the adjacency list for the vertex that was just processed.
        *NOTE*: Do NOT mutate neighbors, it MUST be constant.
"""
Base.@kwdef struct DiskANNCallbacks{A, B, C}
    prequery::A = donothing
    postquery::B = donothing
    postdistance::C = donothing
end

struct StartNode{U,T}
    index::U
    value::T
end

function StartNode(dataset::AbstractVector)
    index = medioid(dataset)
    return StartNode(index, dataset[index])
end

"""
    DiskANNIndex{G,D,S,M}

Index for DiskANN style similarity search.

# Fields and Parameters
* `graph::G` - The relative neighbor graph over the dataset.
* `data::D` - The data set upon which to perform similarity search.
* `startnode::S` - The entry point for queries.
* `metric::M` - Metric to use when performing similarity search.
"""
struct DiskANNIndex{G, D <: AbstractVector, S <: StartNode, M}
    graph::G
    data::D
    startnode::S
    metric::M
end

# constructor
function DiskANNIndex(graph, data::AbstractVector, metric = Euclidean())
    return DiskANNIndex(graph, data, StartNode(data), metric)
end
_Base.Neighbor(::DiskANNIndex, id::T, distance::D) where {T,D} = Neighbor{T,D}(id, distance)

function Base.show(io::IO, index::DiskANNIndex)
    print(io, "DiskANNIndex(", length(index.data), " data points. ")
    print(io, "Entry point index: ", index.startnode.index, ")")
end

struct HasPrefetching end
struct NoPrefetching end

"""
    DiskANNRunner

Collection of pre-allocated data structures for performing querying over a DiskANN index.

# Constructors

    DiskANNRunner(index::DiskANNIndex, search_list_size::Integer; [executor])

Construct a `DiskANNRunner` for `index` with a maximum `search_list_size`.
Increasing `search_list_size` improves the quality of the return results at the cost of
longer query times.

This method also includes an optional `executor` keyword argument. If left at its default
[`single_thread`](@ref), then querying will be single threaded. Pass [`dynamic_thread`](@ref)
to construct a type that will use all available threads for querying.

## Internal Constructors

    DiskANNRunner{I,D}(search_list_size::Integer; [executor])

Construct a `DiskANNRunner` with id-type `I` and distance-type `D` with the given
`search_lists_size` and `executor`. Keyword `executor` defaults to [`single_thread`](@ref).

See also: [`search`](@ref)
"""
mutable struct DiskANNRunner{I <: Integer, D, T <: AbstractSet, P}
    search_list_size::Int64

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
    best::BinaryMinMaxHeap{Neighbor{I,D}}
    best_unvisited::BinaryMinMaxHeap{Neighbor{I,D}}
    visited::T

    # Optional `prefetch_queue`.
    # If `prefetch_queue === nothing` then no prefetching outside of the normal x86 prefetch
    # instructions will be performed.
    #
    # Otherwise, queue must accept vertex IDs.
    prefetch_queue::P
end

_Base.idtype(::DiskANNRunner{I}) where {I} = I
_Base.costtype(::DiskANNRunner{I,D}) where {I,D} = D
function _Base.Neighbor(x::DiskANNRunner, id::Integer, distance)
    I, D = idtype(x), costtype(x)
    # Use `unsafe_trunc` to be slightly faster.
    # In the body of the search routine, we shouldn't see any actual values that will
    # cause the undefined behavior of `unsafe_trunc`.
    return Neighbor{I,D}(unsafe_trunc(I, id), distance)
end

function DiskANNRunner{I,D}(
    search_list_size::Integer;
    executor::F = single_thread,
    prefetch_queue = nothing,
) where {I,D,F}
    best = BinaryMinMaxHeap{Neighbor{I,D}}()
    best_unvisited = BinaryMinMaxHeap{Neighbor{I,D}}()
    visited = RobinSet{I}()
    runner = DiskANNRunner{I, D, typeof(visited), typeof(prefetch_queue)}(
        convert(Int, search_list_size),
        best,
        best_unvisited,
        visited,
        prefetch_queue,
    )

    return threadlocal_wrap(executor, runner)
end

function DiskANNRunner(
    index::DiskANNIndex,
    search_list_size;
    executor::F = single_thread,
    prefetch_queue = nothing,
) where {F}
    I = eltype(index.graph)
    D = costtype(index.metric, index.data)
    return DiskANNRunner{I,D}(search_list_size; executor, prefetch_queue)
end

# Base prefetching on the existence of a non-nothing element in the prefetch queue
hasprefetching(::DiskANNRunner{I,D,T,Nothing}) where {I,D,T} = NoPrefetching()
hasprefetching(::DiskANNRunner{I,D,T,<:Any}) where {I,D,T} = HasPrefetching()

# Prepare for another run.
function Base.empty!(runner::DiskANNRunner)
    empty!(runner.best)
    empty!(runner.best_unvisited)
    empty!(runner.visited)
end

Base.length(runner::DiskANNRunner) = length(runner.best)

visited!(runner::DiskANNRunner, vertex) = push!(runner.visited, getid(vertex))
isvisited(runner::DiskANNRunner, vertex) = in(getid(vertex), runner.visited)
getvisited(runner::DiskANNRunner) = runner.visited

# Get the closest non-visited vertex
# `unsafe_peek` will not remove top element. Unsafe because it assumes queue is nonempty.
unsafe_peek(runner::DiskANNRunner) = @inbounds runner.best_unvisited.valtree[1]
getcandidate!(runner::DiskANNRunner) = getcandidate!(runner, hasprefetching(runner))

# no prefetching case
getcandidate!(runner::DiskANNRunner, ::NoPrefetching) = popmin!(runner.best_unvisited)
# prefetching case
function getcandidate!(runner::DiskANNRunner, ::HasPrefetching)
    @unpack best_unvisited, prefetch_queue = runner
    candidate = popmin!(best_unvisited)

    # If there's another candidate in the queue, try to prefetch it.
    if !isempty(best_unvisited)
        push!(prefetch_queue, getid(unsafe_peek(runner)))
        _Prefetcher.commit!(prefetch_queue)
    end

    return candidate
end

isfull(runner::DiskANNRunner) = length(runner) >= runner.search_list_size

function maybe_pushcandidate!(runner::DiskANNRunner, vertex)
    # If this has already been seen, don't do anything.
    isvisited(runner, vertex) && return nothing
    pushcandidate!(runner, vertex)
end

function pushcandidate!(runner::DiskANNRunner, vertex)
    visited!(runner, vertex)
    @unpack best, best_unvisited = runner

    # Since we have not yet visited this vertex, we have to add it both to `best` and
    # `best_unvisited`,
    push!(best, vertex)
    push!(best_unvisited, vertex)

    # This vertex is a candidate for prefetching if we pushed it right to the front of
    # the queue.
    if getid(unsafe_peek(runner)) == getid(vertex)
        maybe_prefetch(runner, vertex)
    end
    return nothing
end

maybe_prefetch(runner::DiskANNRunner, vertex) = maybe_prefetch(runner, hasprefetching(runner), vertex)
# non-prefetching case
maybe_prefetch(runner::DiskANNRunner, ::NoPrefetching, vertex) = nothing
# prefetching case
function maybe_prefetch(runner::DiskANNRunner, ::HasPrefetching, vertex)
    @unpack prefetch_queue = runner
    push!(prefetch_queue, getid(vertex))
    _Prefetcher.commit!(prefetch_queue)
end

done(runner::DiskANNRunner) = isempty(runner.best_unvisited)
Base.maximum(runner::DiskANNRunner) = _unsafe_maximum(runner.best)

# Bring the size of the best list down to `search_list_size`
# TODO: check if type inference works properly.
# The function `_unsafe_maximum` can return `nothing`, but Julia should be able to
# handle that
function reduce!(runner::DiskANNRunner)
    # Keep ahold of the maximum element in the best_unvisited
    # Since `best_unvisited` is a subset of `best`, we know that this top element lives
    # in `best`.
    # If the element we pull of the top of `best` matches the top of `best_unvisited`, then we
    # need to pop `queue` as well and maintain a new best.
    top = _unsafe_maximum(runner.best_unvisited)
    while length(runner.best) > runner.search_list_size
        vertex = popmax!(runner.best)

        if top !== nothing && getid(vertex) == getid(top)
            popmax!(runner.best_unvisited)
            top = _unsafe_maximum(runner.best_unvisited)
        end
    end
    return nothing
end

"""
    getresults!(runner::DiskANNRunner, num_neighbor) -> AbstractVector

Return the top `num_neighbor` results from `runner`.
"""
function getresults!(runner::DiskANNRunner, num_neighbors)
    return destructive_extract!(runner.best, num_neighbors)
end

#####
##### Greedy Search Implementation
#####

# NOTE: leave `startnode` as an extra `arg` because we override the default behavior of
# unpacking the `index` during the building process.
"""
    search(runner::DiskANNRunner, index::DiskANNIndex, query; [callbacks])

Perform approximate nearest neighbor search for `query` over `index`.
Results can be obtained using [`getresults!`](@ref)

Callbacks is an optional [`DiskANNCallbacks`](@ref) struct to help with gathering metrics.
"""
function _Base.search(
    algo::DiskANNRunner,
    index::DiskANNIndex,
    query::AbstractVector{T},
    start::StartNode = index.startnode;
    callbacks = DiskANNCallbacks(),
    metric = getlocal(index.metric),
) where {T <: Number}
    empty!(algo)

    # Destructure argument
    @unpack graph, data = index
    initial_distance = evaluate(metric, query, start.value)
    pushcandidate!(algo, Neighbor(algo, start.index, initial_distance))

    @inbounds while !done(algo)
        p = getid(unsafe_peek(algo))
        neighbors = LightGraphs.outneighbors(graph, p)

        # Prefetch all new datapoints.
        # IMPORTANT: This is critical for performance!
        @inbounds for vertex in neighbors
            prefetch(data, vertex)
        end

        # Prune
        # Do this here to allow the prefetched vectors time to arrive in the cache.
        getcandidate!(algo)
        reduce!(algo)
        algmax = getdistance(maximum(algo))

        # Distance computations
        for v in neighbors
            # Perform distance query, and try to prefetch the next datapoint.
            # NOTE: Checking if a vertex has been visited here is actually SLOWER than
            # deferring until after the distance comparison.
            @inbounds d = evaluate(metric, query, pointer(data, v))

            ## only bother to add if it's better than the worst currently tracked.
            if d < algmax || !isfull(algo)
                maybe_pushcandidate!(algo, Neighbor(algo, v, d))
            end
        end

        callbacks.postdistance(algo, p, neighbors)
    end

    return nothing
end

"""
    search(runner::DiskANNRunner, index::DiskANNIndex, queries; [num_neighbors], [callbacks])

Perform approximate nearest neighbor search for all entries in `queries`, returning
`num_neighbors` results for each query. The results is a `num_neighbors × length(queries)`
matrix. The nearest neighbors for `queries[i]` will be in column `i` of the returned
matrix, sorted from nearest to furthest.

Callbacks is an optional [`DiskANNCallbacks`](@ref) struct to help with gathering metrics.
"""
function _Base.search(
    runner::MaybeThreadLocal{DiskANNRunner},
    index::DiskANNIndex,
    queries::AbstractVector{T};
    num_neighbors = 10,
    kw...,
) where {T <: AbstractVector}
    num_queries = length(queries)
    dest = Array{eltype(index.graph),2}(undef, num_neighbors, num_queries)
    return search!(dest, runner, index, queries; num_neighbors, kw...)
end

# Single Threaded Query
function _Base.search!(
    dest::AbstractMatrix,
    algo::DiskANNRunner,
    index::DiskANNIndex,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = DiskANNCallbacks(),
)
    for (col, query) in enumerate(queries)
        # -- optional telemetry
        callbacks.prequery()

        _Base.prehook(getlocal(index.metric), query)
        search(algo, index, query; callbacks)

        # Copy over the results to the destination
        results = getresults!(algo, num_neighbors)
        for i in 1:num_neighbors
            @inbounds dest[i,col] = getid(results[i])
        end

        # -- optional telemetry
        callbacks.postquery()
    end
    return dest
end


# Multi Threaded Query
function _Base.search!(
    dest::AbstractMatrix,
    tls::ThreadLocal{<:DiskANNRunner},
    index::DiskANNIndex,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = DiskANNCallbacks(),
)
    dynamic_thread(getpool(tls), eachindex(queries), 8) do col
        #_metric = _Base.distribute_distance(metric)
        query = queries[col]
        algo = tls[]

        # -- optional telemetry
        callbacks.prequery()

        _Base.prehook(getlocal(index.metric), query)
        search(algo, index, query; callbacks = callbacks)

        # Copy over the results to the destination
        results = getresults!(algo, num_neighbors)
        for i in 1:num_neighbors
            @inbounds dest[i,col] = getid(results[i])
        end

        # -- optional telemetry
        callbacks.postquery()
    end
    return dest
end


#####
##### Callback Implementations
#####

struct Latencies{T <: MaybeThreadLocal{Vector{UInt64}}}
    values::T
end

Base.empty!(x::Latencies{Vector{UInt64}}) = empty!(x.values)
Base.empty!(x::Latencies{<:ThreadLocal}) = foreach(empty!, getall(x.values))

Base.getindex(x::Latencies{Vector{UInt64}}) = x.values
Base.getindex(x::Latencies{<:ThreadLocal}) = x.values[]

Base.get(x::Latencies{Vector{UInt64}}) = x.values
Base.get(x::Latencies{<:ThreadLocal}) = reduce(vcat, getall(x.values))

Latencies(::DiskANNRunner) = Latencies(UInt64[])
Latencies(::ThreadLocal{<:DiskANNRunner}) = Latencies(ThreadLocal(UInt64[]))

# single threaded version
function latency_callbacks(runner::MaybeThreadLocal{DiskANNRunner})
    latencies = Latencies(runner)
    prequery = () -> push!(latencies[], time_ns())
    postquery = () -> begin
        _latencies = latencies[]
        _latencies[end] = time_ns() - _latencies[end]
    end
    return (latencies = latencies, callbacks = DiskANNCallbacks(; prequery, postquery))
end

function visited_callbacks()
    histogram = zeros(Int, 1_000_000_000)
    postdistance = function __postdistance(_, _, neighbors)
        for v in neighbors
            histogram[v] += 1
        end
    end
    return (histogram = histogram, callbacks = DiskANNCallbacks(; postdistance))
end

