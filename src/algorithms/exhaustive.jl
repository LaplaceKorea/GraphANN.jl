#####
##### Exhaustive
#####

raw"""
    exhaustive_search(queries, dataset, [num_neighbors]; [groupsize]) -> Matrix{UInt64}

Perform a exhaustive nearest search operation of `queries` on `dataset`, finding the closest
`num_neighbors` neighbors.

Returns a `Matrix{UInt64}` where column `i` lists the ids of the `num_neighbors` nearest
neighbors of `query[i]` in descending order (i.e., index (i,1) is the nearest neighbor,
(i,2) is the second nearest etc.)

## Performance Quidelines

This process will go much faster if all the available threads on the machine are used.
I.E., `export JULIA_NUM_THREADS=$(nproc)`

The exhaustive search is implemented by grouping queries into groups of `groupsize` and
then scaning the entire dataset to find the nearest neighbors for each query in the group.

Making this number too small will result in extremely excessive memory traffic, while making
it too large will hurt performance in the cache.

The default is `groupsize = 32`.
"""
function exhaustive_search(
    queries::AbstractVector{A},
    dataset::AbstractVector{B},
    num_neighbors::Integer = 100;
    idtype::Type{I} = UInt32,
    costtype::Type{D} = costtype(A, B),
    executor = dynamic_thread,
    groupsize = 32,
    kw...
) where {A,B,I,D}
    runner = ExhaustiveRunner(
        I,
        length(queries),
        num_neighbors;
        executor = executor,
        costtype = D,
        max_groupsize = groupsize,
    )
    search!(runner, queries, dataset; groupsize, kw...)
    return runner.groundtruth
end

const VecOrMat{T} = Union{AbstractVector{T}, AbstractMatrix{T}}
const IntOrNeighbor{I} = Union{I, <:Neighbor{I}}
inttype(::Type{I}) where {I <: Integer} = I
inttype(::Type{<:Neighbor{I}}) where {I <: Integer} = I

mutable struct ExhaustiveRunner{V <: VecOrMat{<:IntOrNeighbor}, T <: MaybeThreadLocal{<:AbstractVector}, F}
    groundtruth::V
    exhaustive_local::T
    executor::F
end

function ExhaustiveRunner(
    ::Type{ID},
    num_queries::Integer,
    num_neighbors::U = one;
    executor::F = single_thread,
    costtype::Type{D} = Float32,
    max_groupsize = 32,
) where {ID <: IntOrNeighbor, U <: Union{Integer, typeof(one)}, F, D}
    # Preallocate destination.
    I = inttype(ID)
    if isa(num_neighbors, Integer)
        groundtruth = Matrix{I}(undef, num_neighbors, num_queries)
    else
        groundtruth = Vector{I}(undef, num_queries)
    end

    # Create thread local storage.
    heaps = [BoundedMaxHeap{Neighbor{I,D}}(_num_neighbors(num_neighbors)) for _ in 1:max_groupsize]
    exhaustive_local = threadlocal_wrap(executor, heaps)
    return ExhaustiveRunner(groundtruth, exhaustive_local, executor)
end

threadlocal_wrap(::typeof(dynamic_thread), heaps) = ThreadLocal(heaps)
threadlocal_wrap(::typeof(single_thread), heaps) = heaps

_num_neighbors(::AbstractVector) = 1
_num_neighbors(x::AbstractMatrix) = size(x, 1)
_num_neighbors(::typeof(one)) = 1
_num_neighbors(x::Integer) = x

# Allow resizing when using a vector to store the single nearest neighbor.
function Base.resize!(runner::ExhaustiveRunner{<:AbstractVector}, sz::Integer)
    resize!(runner.groundtruth, sz)
end

function sizecheck(groundtruth::AbstractVector, num_neighbors, num_queries)
    @assert num_neighbors == 1
    @assert num_queries == length(groundtruth)
end

function sizecheck(groundtruth::AbstractMatrix, num_neighbors, num_queries)
    @assert size(groundtruth) == (num_neighbors, num_queries)
end

function search!(
    runner::ExhaustiveRunner,
    queries::AbstractVector,
    dataset::AbstractVector;
    groupsize = 32,
    meter = ProgressMeter.Progress(length(queries), 1),
    metric = Euclidean(),
)
    # Destructure runner
    @unpack groundtruth, exhaustive_local, executor = runner
    num_neighbors = _num_neighbors(groundtruth)
    num_queries = length(queries)

    sizecheck(groundtruth, num_neighbors, num_queries)

    # Batch the query range so each thread works on a chunk of queries at a time.
    # The dynamic load balancer will give one batch at a time to each worker.
    batched_iter = BatchedRange(1:num_queries, groupsize)

    # N.B. Need to use the "let" trick on `exhaustive_local` since it is captured in the
    # closure passed to the `executor`.
    #
    # If we don't, then the `_nearest_neighbors!` and `_commit!` inner functions require
    # dynamic dispatch.
    let exhaustive_local = exhaustive_local
        executor(batched_iter) do range
            heaps = _Base.getlocal(exhaustive_local)

            # Compute nearest neighbors for this batch across the whole dataset.
            # Implement this as an inner function to help out type inference.
            _nearest_neighbors!(heaps, dataset, queries, range, metric)
            _commit!(groundtruth, heaps, range)

            # Note: ProgressMeter is threadsafe - so calling it here is okay.
            # With modestly sized dataset, the arrival time of threads to this point should be
            # staggered enough that lock contention shouldn't be an issue.
            #
            # For reference, it takes ~200 seconds to compute 100 nearest neighbors for a
            # batch of 32 queries on Sift100M.
            #
            # That is FAR longer than any time will be spent fighting over this lock.
            meter !== nothing && ProgressMeter.next!(meter; step = length(range))
        end
    end
    meter !== nothing && ProgressMeter.finish!(meter)
    return groundtruth
end

#####
##### Inner functions
#####

# Define there to help out inference in the "exhaustive_search!" closure.
function _nearest_neighbors!(
    heaps::AbstractVector{BoundedMaxHeap{T}},
    dataset::AbstractVector,
    queries::AbstractVector,
    range,
    metric,
) where {T <: Neighbor}
    for (base_id, base) in enumerate(dataset)
        for (heap_num, query_id) in enumerate(range)
            query = queries[query_id]
            dist = evaluate(metric, query, base)

            # Need to convert from 1 based indexing to 0 based indexing...
            push!(heaps[heap_num], T(base_id - 1, dist))
        end
    end
    return nothing
end

# Populate destinations based on eltype (i.e., `Integer` or `Neighbor`).
# Matrix Version
_populate!(gt::AbstractVector{<:Integer}, heap) = gt .= getid.(destructive_extract!(heap))
_populate!(gt::AbstractVector{<:Neighbor}, heap) = gt .= destructive_extract!(heap)
function _commit!(gt::AbstractMatrix, heaps::AbstractVector, range)
    for (heap_num, query_id) in enumerate(range)
        heap = heaps[heap_num]
        _populate!(view(gt, :, query_id), heap)
        empty!(heap)
    end
    return nothing
end

# Vector Version
_set!(x::AbstractVector{<:Integer}, v::Neighbor, i::Integer) = x[i] = getid(v)
_set!(x::AbstractVector{<:Neighbor}, v::Neighbor, i::Integer) = x[i] = v
function _commit!(gt::AbstractVector, heaps::AbstractVector, range)
    for (heap_num, query_id) in enumerate(range)
        heap = heaps[heap_num]
        id = only(destructive_extract!(heap))
        _set!(gt, id, query_id)
        empty!(heap)
    end
    return nothing
end

