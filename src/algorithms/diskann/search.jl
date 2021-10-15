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
    Signature: `postdistance(runner::DiskANNRunner, vertex::Integer, neighbors::AbstractVector)`.
    - `runner` provides the current state of the search.
    - `neighbors` the adjacency list for the vertex that was just processed.
        *NOTE*: Do NOT mutate neighbors, it MUST be constant.
"""
Base.@kwdef struct DiskANNCallbacks{A,B,C}
    prequery::A = donothing
    postquery::B = donothing
    postdistance::C = donothing
end

struct StartNode{U,T}
    index::U
    value::T
end

function StartNode(dataset::AbstractVector; metric = Euclidean())
    index = medioid(dataset; metric)
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
struct DiskANNIndex{G,D<:AbstractVector,S<:StartNode,M}
    graph::G
    data::D
    startnode::S
    metric::M
end

# constructor
_forward(_, x::StartNode) = x
_forward(data, index::Integer) = StartNode(index, data[index])
function DiskANNIndex(
    graph, data::AbstractVector, metric = Euclidean(); startnode = StartNode(data; metric)
)
    return DiskANNIndex(graph, data, _forward(data, startnode), metric)
end
_Base.Neighbor(::DiskANNIndex, id::T, distance::D) where {T,D} = Neighbor{T,D}(id, distance)

function Base.show(io::IO, index::DiskANNIndex)
    print(io, "DiskANNIndex(", length(index.data), " data points. ")
    return print(io, "Entry point index: ", index.startnode.index, ")")
end

abstract type AbstractDiskANNRunner{I<:Integer,D,O<:Base.Ordering} end

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
mutable struct DiskANNRunner{I,D,O} <: AbstractDiskANNRunner{I,D,O}
    search_list_size::Int64

    # Pre-allocated buffer for the search list
    buffer::BestBuffer{DistanceLSB,I,D,O}
    visited::FastSet{I}
end

_Base.idtype(::AbstractDiskANNRunner{I}) where {I} = I
_Base.costtype(::AbstractDiskANNRunner{I,D}) where {I,D} = D

function _Base.Neighbor(::AbstractDiskANNRunner{I,D}, id::Integer, distance) where {I,D}
    # Use `unsafe_trunc` to be slightly faster.
    # In the body of the search routine, we shouldn't see any actual values that will
    # cause the undefined behavior of `unsafe_trunc`.
    return Neighbor{I,D}(unsafe_trunc(I, id), distance)
end

function DiskANNRunner{I,D}(
    search_list_size::Integer, ordering::O; executor::F = single_thread
) where {I,D,F,O<:Base.Ordering}
    buffer = BestBuffer{DistanceLSB,I,D}(search_list_size, ordering)
    visited = FastSet{I}()
    runner = DiskANNRunner{I,D,O}(convert(Int, search_list_size), buffer, visited)
    return threadlocal_wrap(executor, runner)
end
@inline Base.lt(o::AbstractDiskANNRunner, x, y) = Base.lt(o.buffer, x, y)

function Base.resize!(runner::AbstractDiskANNRunner, val::Integer)
    runner.search_list_size = val
    return resize!(runner.buffer, val)
end

function Base.resize!(runner::ThreadLocal{<:DiskANNRunner}, val::Integer)
    return foreach(x -> resize!(x, val), getall(runner))
end

function DiskANNRunner(
    index::DiskANNIndex, search_list_size; executor::F = single_thread
) where {F,U}
    I = eltype(index.graph)
    D = costtype(index.metric, index.data)
    return DiskANNRunner{I,D}(search_list_size, ordering(index.metric); executor)
end

# Prepare for another run.
function Base.empty!(runner::AbstractDiskANNRunner)
    empty!(runner.buffer)
    return empty!(runner.visited)
end

Base.length(runner::AbstractDiskANNRunner) = length(runner.buffer)

visited!(runner::AbstractDiskANNRunner, vertex) = push!(runner.visited, getid(vertex))
isvisited(runner::AbstractDiskANNRunner, vertex) = in(getid(vertex), runner.visited)
getvisited(runner::AbstractDiskANNRunner) = runner.visited

# Get the closest non-visited vertex
# `unsafe_peek` will not remove top element. Unsafe because it assumes queue is nonempty.
function unsafe_peek(runner::AbstractDiskANNRunner)
    return runner.buffer.entries[runner.buffer.bestunvisited]
end
getcandidate!(runner::AbstractDiskANNRunner) = getcandidate!(runner.buffer)
isfull(runner::AbstractDiskANNRunner) = length(runner) >= runner.search_list_size

function maybe_pushcandidate!(runner::AbstractDiskANNRunner, vertex::Neighbor)
    # If this has already been seen, don't do anything.
    isvisited(runner, vertex) && return false
    return pushcandidate!(runner, vertex)
end

function pushcandidate!(runner::AbstractDiskANNRunner, vertex::Neighbor)
    visited!(runner, vertex)
    @unpack buffer = runner

    # Insert into queue
    return insert!(buffer, vertex)
end

done(runner::AbstractDiskANNRunner) = done(runner.buffer)
Base.maximum(runner::AbstractDiskANNRunner) = maximum(runner.buffer)

"""
    getresults!(runner::AbstractDiskANNRunner, num_neighbor, [query]) -> AbstractVector

Return the top `num_neighbor` results from `runner`.
"""
function getresults!(runner::AbstractDiskANNRunner, num_neighbors, query)
    return view(runner.buffer.entries, Base.OneTo(num_neighbors))
end

#####
##### Greedy Search Implementation
#####

# NOTE: leave `startnode` as an extra `arg` because we override the default behavior of
# unpacking the `index` during the building process.
"""
    search(runner::AbstractDiskANNRunner, index::DiskANNIndex, query; [callbacks])

Perform approximate nearest neighbor search for `query` over `index`.
Results can be obtained using [`getresults!`](@ref)

Callbacks is an optional [`DiskANNCallbacks`](@ref) struct to help with gathering metrics.
"""
function _Base.search(
    runner::AbstractDiskANNRunner,
    index::DiskANNIndex,
    query::MaybePtr{AbstractVector{T}},
    start::StartNode = index.startnode;
    callbacks = DiskANNCallbacks(),
    metric = getlocal(index.metric),
) where {T<:Number}
    empty!(runner)

    # Destructure argument
    @unpack graph, data = index
    initial_distance = evaluate(metric, query, start.value)
    pushcandidate!(runner, Neighbor(runner, start.index, initial_distance))

    @inbounds while !done(runner)
        p = getid(unsafe_peek(runner))
        neighbors = LightGraphs.outneighbors(graph, p)

        # Prefetch all new datapoints.
        # IMPORTANT: This is critical for performance!
        @inbounds for vertex in neighbors
            prefetch(data, vertex)
        end

        # Prune
        # Do this here to allow the prefetched vectors time to arrive in the cache.
        getcandidate!(runner)
        algmax = getdistance(maximum(runner))

        # Prefetch potential next neigbors.
        !done(runner) && unsafe_prefetch(graph, getid(unsafe_peek(runner)))

        # Distance computations
        for v in neighbors
            # Perform distance query, and try to trefetch the next datapoint.
            # NOTE: Checking if a vertex has been visited here is actually SLOWER than
            # deferring until after the distance comparison.
            @inbounds d = evaluate(metric, query, pointer(data, v))

            ## only bother to add if it's better than the worst currently tracked.
            if Base.lt(runner, d, algmax) || !isfull(runner)
                maybe_pushcandidate!(runner, Neighbor(runner, v, d))
                algmax = getdistance(maximum(runner))
            end
        end

        callbacks.postdistance(runner, p, neighbors)
    end

    return nothing
end

"""
    search(runner::AbstractDiskANNRunner, index::DiskANNIndex, queries; [num_neighbors], [callbacks], [postprocess!])

Perform approximate nearest neighbor search for all entries in `queries`, returning
`num_neighbors` results for each query. The results is a `num_neighbors × length(queries)`
matrix. The nearest neighbors for `queries[i]` will be in column `i` of the returned
matrix, sorted from nearest to furthest.

Callbacks is an optional [`DiskANNCallbacks`](@ref) struct to help with gathering metrics.

Post Processing
---------------

The keyword argument `postprocess!` lets users to apply their own extraction methods
for data located in the `runner`, allowing for operations like final reranking when
quantization is being used. This function is allowed to modify the state of `runner` and
leave it in a potentially illegal state.

The default implementation is `GraphANN._Base.getresults!`. Take inspiration from the
implementation of that function when defining your own.
"""
function _Base.search(
    runner::MaybeThreadLocal{AbstractDiskANNRunner},
    index::DiskANNIndex,
    queries::AbstractVector{T};
    num_neighbors = 10,
    kw...,
) where {T<:AbstractVector}
    num_queries = length(queries)
    dest = Array{eltype(index.graph),2}(undef, num_neighbors, num_queries)
    return search!(dest, runner, index, queries; num_neighbors, kw...)
end

# Single Threaded Query
function _Base.search!(
    dest::AbstractMatrix,
    runner::AbstractDiskANNRunner,
    index::DiskANNIndex,
    queries::AbstractVector{T};
    num_neighbors = 10,
    callbacks = DiskANNCallbacks(),
    postprocess!::F = getresults!,
) where {T<:AbstractVector,F}
    metric = getlocal(index.metric)
    for col in eachindex(queries)
        query = pointer(queries, col)
        # -- optional telemetry
        callbacks.prequery()

        _Base.prehook(metric, query)
        search(runner, index, query; callbacks)

        # Copy over the results to the destination
        results = postprocess!(runner, num_neighbors, query)
        for i in 1:num_neighbors
            @inbounds dest[i, col] = getid(results[i])
        end

        # -- optional telemetry
        callbacks.postquery()
    end
    return dest
end

# Multi Threaded Query
function _Base.search!(
    dest::AbstractMatrix,
    tls::ThreadLocal{<:AbstractDiskANNRunner},
    index::DiskANNIndex,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = DiskANNCallbacks(),
    postprocess!::F = getresults!,
) where {F}
    dynamic_thread(getpool(tls), eachindex(queries), 64) do col
        #_metric = _Base.distribute_distance(metric)
        metric = getlocal(index.metric)
        query = pointer(queries, col)
        runner = tls[]

        # -- optional telemetry
        callbacks.prequery()

        _Base.prehook(metric, query)
        search(runner, index, query; callbacks = callbacks)

        # Copy over the results to the destination
        results = postprocess!(runner, num_neighbors, query)
        for i in 1:num_neighbors
            @inbounds dest[i, col] = getid(results[i])
        end

        # -- optional telemetry
        callbacks.postquery()
    end
    return dest
end

#####
##### DiskANN Callbacks
#####

function visited_callbacks(data)
    histogram = zeros(Int, length(data))
    postdistance = function __postdistance(_, _, neighbors)
        for v in neighbors
            histogram[v] += 1
        end
    end
    return (histogram = histogram, callbacks = DiskANNCallbacks(; postdistance))
end
