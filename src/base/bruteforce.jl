#####
##### BruteForce
#####

raw"""
    bruteforce_search(queries, dataset, [num_neighbors]; [groupsize]) -> Matrix{UInt64}

Perform a bruteforce nearest search operation of `queries` on `dataset`, finding the closest
`num_neighbors` neighbors.

Returns a `Matrix{UInt64}` where column `i` lists the ids of the `num_neighbors` nearest
neighbors of `query[i]` in descending order (i.e., index (i,1) is the nearest neighbor,
(i,2) is the second nearest etc.)

## Performance Quidelines

This process will go much faster if all the available threads on the machine are used.
I.E., `export JULIA_NUM_THREADS=$(nproc)`

The bruteforce search is implemented by grouping queries into groups of `groupsize` and
then scaning the entire dataset to find the nearest neighbors for each query in the group.

Making this number too small will result in extremely excessive memory traffic, while making
it too large will hurt performance in the cache.

The default is `groupsize = 32`.
"""
function bruteforce_search(
    queries::AbstractVector,
    dataset::AbstractVector,
    num_neighbors::Integer = 100;
    idtype::Type{I} = UInt32,
    kw...
) where {I}
    gt = Array{I,2}(undef, num_neighbors, length(queries))
    bruteforce_search!(gt, queries, dataset; idtype = I, kw...)
    return gt
end

const VecOrMat{T} = Union{AbstractVector{T}, AbstractMatrix{T}}

_num_neighbors(::AbstractVector) = 1
_num_neighbors(x::AbstractMatrix) = size(x, 1)

function bruteforce_search!(
    gt::VecOrMat{ID},
    queries::AbstractVector{A},
    dataset::AbstractVector{B};
    executor::F = dynamic_thread,
    groupsize = 32,
    meter = ProgressMeter.Progress(length(queries), 1),
    idtype::Type{I} = UInt32,
    costtype::Type{D} = costtype(A, B),
    metric = Euclidean(),
    tls::Union{Nothing, <:ThreadLocal, <:AbstractVector} = nothing,
) where {A, B, I, D, ID <: Union{I, Neighbor{I, D}}, F}
    num_neighbors = _num_neighbors(gt)
    num_queries = length(queries)

    # Allocate TLS if not supplied by the caller.
    if tls === nothing
        tls = bruteforce_threadlocal(executor, I, D, num_neighbors, groupsize)
    end

    # Batch the query range so each thread works on a chunk of queries at a time.
    # The dynamic load balancer will give one batch at a time to each worker.
    batched_iter = BatchedRange(1:num_queries, groupsize)

    # N.B. Need to use the "let" trick on `tls` since it is captured in the closure passed
    # to the `executor`.
    #
    # If we don't, then the `_nearest_neighbors!` and `_commit!` inner functions require
    # dynamic dispatch.
    let tls = tls
        executor(batched_iter) do range
            heaps = getlocal(tls)

            # Compute nearest neighbors for this batch across the whole dataset.
            # Implement this as an inner function to help out type inference.
            _nearest_neighbors!(heaps, dataset, queries, range, metric)
            _commit!(gt, heaps, range)

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
    return gt
end

threadlocal_wrap(::typeof(dynamic_thread), heaps) = ThreadLocal(heaps)
threadlocal_wrap(::typeof(single_thread), heaps) = heaps
function bruteforce_threadlocal(f::F, ::Type{I}, ::Type{D}, num_neighbors, groupsize) where {F,I,D}
    heaps = [BoundedMaxHeap{Neighbor{I,D}}(num_neighbors) for _ in 1:groupsize]
    return threadlocal_wrap(f, heaps)
end

#####
##### Inner bruteforce functions
#####

# Define there to help out inference in the loop body
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

