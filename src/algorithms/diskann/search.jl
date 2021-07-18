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
Base.@kwdef struct DiskANNCallbacks{A,B,C}
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
struct DiskANNIndex{G,D<:AbstractVector,S<:StartNode,M}
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
    return print(io, "Entry point index: ", index.startnode.index, ")")
end

# Wrapper for Neighbors that uses the LSB of the distance field to track if the
# neighbor has been expanded/visited yet.
struct MaskWrap{I,D}
    neighbor::Neighbor{I,D}
    MaskWrap{I,D}(neighbor::Neighbor{I,D}) where {I,D} = new{I,D}(neighbor)
end

function MaskWrap(x::Neighbor{I,D}) where {I,D}
    return MaskWrap{I,D}(Neighbor{I,D}(x.id, clearlsb(x.distance)))
end

_Base.getid(x::MaskWrap) = getid(x.neighbor)
_Base.getdistance(x::MaskWrap) = clearlsb(x.neighbor.distance)
unwrap(x::MaskWrap) = x.neighbor

isvisited(x::MaskWrap) = isone(getlsb(x))

getlsb(x::T) where {T <: Integer} = x & one(T)
getlsb(x::Float32) = getlsb(reinterpret(UInt32, x))
getlsb(x::Float64) = getlsb(reinterpret(UInt64, x))
getlsb(x::MaskWrap) = getlsb(x.neighbor.distance)

clearlsb(x::T) where {T <: Integer} = x & ~(one(T))
clearlsb(x::Float32) = reinterpret(Float32, clearlsb(reinterpret(UInt32, x)))
clearlsb(x::Float64) = reinterpret(Float64, clearlsb(reinterpret(UInt64, x)))
function clearlsb(x::MaskWrap{I,D}) where {I,D}
    @unpack id, distance = x.neighbor
    return MaskWrap{I,D}(Neighbor{I,D}(id, clearlsb(distance)))
end

setlsb(x::T) where {T <: Integer} = x | one(T)
setlsb(x::Float32) = reinterpret(Float32, setlsb(reinterpret(UInt32, x)))
setlsb(x::Float64) = reinterpret(Float64, setlsb(reinterpret(UInt64, x)))
function setlsb(x::MaskWrap{I,D}) where {I,D}
    @unpack id, distance = x.neighbor
    return MaskWrap{I,D}(Neighbor{I,D}(id, setlsb(distance)))
end

function Base.isless(x::MaskWrap{I,D}, y::MaskWrap{I,D}) where {I,D}
    # use "<" to generate smaller code for floats
    return getdistance(x) < getdistance(y)
end

#####
##### BestBuffer
#####

mutable struct BestBuffer{I,D}
    entries::Vector{MaskWrap{I,D}}
    # How many entries are currently valid in the vector.
    currentlength::Int
    maxlength::Int
    # The index of the lowest-distance entry that has not yet been visited.
    bestunvisited::Int
end

function BestBuffer{I,D}(maxlen::Integer) where {I,D}
    entries = Vector{MaskWrap{I,D}}(undef, maxlen)
    return BestBuffer{I,D}(entries, 0, maxlen, 1)
end

function Base.empty!(buffer::BestBuffer)
    buffer.currentlength = 0
    buffer.bestunvisited = 1
    return nothing
end

function Base.resize!(buffer::BestBuffer, val)
    resize!(buffer.entries, val)
    buffer.maxlength = val
    return nothing
end

Base.length(x::BestBuffer) = x.currentlength
Base.maximum(x::BestBuffer) = unwrap(x.entries[x.currentlength])

Base.insert!(x::BestBuffer, v::Neighbor) = insert!(x, MaskWrap(v))
function Base.insert!(x::BestBuffer, v::MaskWrap)
    @unpack entries, currentlength, bestunvisited, maxlength = x
    i = 1
    while i <= currentlength
        isless(v, @inbounds(entries[i])) && break
        i += 1
    end
    i > maxlength && return nothing

    shift!(x, i)
    @inbounds entries[i] = v

    # Update best unvisited.
    if i < bestunvisited
        x.bestunvisited = i
    end
    return nothing
end

function shift!(x::BestBuffer, i)
    @unpack entries, currentlength, maxlength = x
    droplast = currentlength == maxlength
    if droplast
        maxind = (maxlength - 1)
    else
        maxind = currentlength
        x.currentlength = currentlength + 1
    end

    for j in (maxlength-1):-1:i
        @inbounds(entries[j+1] = entries[j])
    end

    return nothing
end

function getcandidate!(x::BestBuffer)
    @unpack entries, bestunvisited, currentlength = x
    candidate = entries[bestunvisited]
    entries[bestunvisited] = setlsb(candidate)

    # Find the next unvisited candidate.
    i = bestunvisited + 1
    while i <= currentlength
        !isvisited(@inbounds(entries[i])) && break
        i += 1
    end
    x.bestunvisited = i
    return unwrap(candidate)
end

done(x::BestBuffer) = (x.bestunvisited > x.currentlength)

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
mutable struct DiskANNRunner{I<:Integer,D,T<:AbstractSet}
    search_list_size::Int64

    # Pre-allocated buffer for the search list
    buffer::BestBuffer{I,D}
    visited::T
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
    search_list_size::Integer; executor::F = single_thread,
) where {I,D,F}
    buffer = BestBuffer{I,D}(search_list_size)
    visited = RobinSet{I}()
    runner = DiskANNRunner{I,D,typeof(visited)}(
        convert(Int, search_list_size), buffer, visited,
    )

    return threadlocal_wrap(executor, runner)
end

function Base.resize!(runner::DiskANNRunner, val::Integer)
    runner.search_list_size = val
    resize!(runner.buffer, val)
end

function Base.resize!(runner::ThreadLocal{<:DiskANNRunner}, val::Integer)
    return foreach(x -> resize!(x, val), getall(runner))
end

function DiskANNRunner(
    index::DiskANNIndex,
    search_list_size;
    executor::F = single_thread,
) where {F}
    I = eltype(index.graph)
    D = costtype(index.metric, index.data)
    return DiskANNRunner{I,D}(search_list_size; executor)
end

# Prepare for another run.
function Base.empty!(runner::DiskANNRunner)
    empty!(runner.buffer)
    return empty!(runner.visited)
end

Base.length(runner::DiskANNRunner) = length(runner.buffer)
visited!(runner::DiskANNRunner, vertex) = push!(runner.visited, getid(vertex))
isvisited(runner::DiskANNRunner, vertex) = in(getid(vertex), runner.visited)
getvisited(runner::DiskANNRunner) = runner.visited

# Get the closest non-visited vertex
# `unsafe_peek` will not remove top element. Unsafe because it assumes queue is nonempty.
unsafe_peek(runner::DiskANNRunner) = runner.buffer.entries[runner.buffer.bestunvisited]
getcandidate!(runner::DiskANNRunner) = getcandidate!(runner.buffer)
isfull(runner::DiskANNRunner) = length(runner) >= runner.search_list_size

function maybe_pushcandidate!(runner::DiskANNRunner, vertex)
    # If this has already been seen, don't do anything.
    isvisited(runner, vertex) && return nothing
    return pushcandidate!(runner, vertex)
end

function pushcandidate!(runner::DiskANNRunner, vertex)
    visited!(runner, vertex)
    @unpack buffer = runner

    # Insert into queue
    insert!(buffer, vertex)
    return nothing
end

done(runner::DiskANNRunner) = done(runner.buffer)
Base.maximum(runner::DiskANNRunner) = maximum(runner.buffer)

"""
    getresults!(runner::DiskANNRunner, num_neighbor) -> AbstractVector

Return the top `num_neighbor` results from `runner`.
"""
function getresults!(runner::DiskANNRunner, num_neighbors)
    results = view(runner.buffer.entries, Base.OneTo(num_neighbors))
    return results
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
    query::MaybePtr{AbstractVector{T}},
    start::StartNode = index.startnode;
    callbacks = DiskANNCallbacks(),
    metric = getlocal(index.metric),
) where {T<:Number}
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
        algmax = getdistance(maximum(algo))

        # Distance computations
        for v in neighbors
            # Perform distance query, and try to trefetch the next datapoint.
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
) where {T<:AbstractVector}
    num_queries = length(queries)
    dest = Array{eltype(index.graph),2}(undef, num_neighbors, num_queries)
    return search!(dest, runner, index, queries; num_neighbors, kw...)
end

# Single Threaded Query
function _Base.search!(
    dest::AbstractMatrix,
    algo::DiskANNRunner,
    index::DiskANNIndex,
    queries::AbstractVector{T};
    num_neighbors = 10,
    callbacks = DiskANNCallbacks(),
) where {T <: AbstractVector}
    for col in eachindex(queries)
        query = pointer(queries, col)
        # -- optional telemetry
        callbacks.prequery()

        _Base.prehook(getlocal(index.metric), query)
        search(algo, index, query; callbacks)

        # Copy over the results to the destination
        results = getresults!(algo, num_neighbors)
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
    tls::ThreadLocal{<:DiskANNRunner},
    index::DiskANNIndex,
    queries::AbstractVector;
    num_neighbors = 10,
    callbacks = DiskANNCallbacks(),
)
    dynamic_thread(getpool(tls), eachindex(queries), 64) do col
        #_metric = _Base.distribute_distance(metric)
        query = pointer(queries, col)
        algo = tls[]

        # -- optional telemetry
        callbacks.prequery()

        _Base.prehook(getlocal(index.metric), query)
        search(algo, index, query; callbacks = callbacks)

        # Copy over the results to the destination
        results = getresults!(algo, num_neighbors)
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

function visited_callbacks()
    histogram = zeros(Int, 1_000_000_000)
    postdistance = function __postdistance(_, _, neighbors)
        for v in neighbors
            histogram[v] += 1
        end
    end
    return (histogram = histogram, callbacks = DiskANNCallbacks(; postdistance))
end
