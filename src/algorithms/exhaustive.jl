#####
##### Exhaustive Search
#####

"""
    exhaustive_search(queries, dataset, [num_neighbors]; [metric, idtype, groupsize]) -> Matrix{idtype}

Perform a exhaustive nearest search operation of `queries` on `dataset`, finding the closest
`num_neighbors` neighbors with respect to `metric`.

Returns a `Matrix{idtype}` where column `i` lists the ids of the `num_neighbors` nearest
neighbors of `query[i]` in descending order (i.e., index (i,1) is the nearest neighbor,
(i,2) is the second nearest etc.)

**Note**: Nearest neighbors will converted to index-0.

## Example

```jldoctest
julia> dataset = GraphANN.sample_dataset();

julia> queries = GraphANN.sample_queries();

julia> neighbors = GraphANN.exhaustive_search(queries, dataset, 10)
10×100 Matrix{Int64}:
 2176  2781  2707  9843  4719  1097  …  8419  9205  7512  8825  5460  8082
 3752  9574  9938  9825  5164  1239     8099  1185  5332  9081  5439  8782
  882  2492  2698  9574  1671  4943     7678  9682  7939  6142  5810  4767
 4009  1322  9972  9582  1538  3227     7486  7271   266  5072  6882  9163
 2837  3136  6995  4097  5897   804     8608  2265  2126  6234  5773  9942
  190  1038  6801  9576  4764  2607  …  8492  3055  4867  7467  6906  9993
 3615  9564  8906  9581  4559  4060     6804  1130  3206  6926  5924  9240
  816   925  5232   272   358  4443     7411   349  2728  8102  6745  2894
 1045  3998  6162  9575  5775  4246     9267  2318  2949  5134  7419  9338
 1884  2183  5199  4096  4622  3112     4053   925  8307  8471  6047  9846
```

## Performance Quidelines

This process will go much faster if all the available threads on the machine are used.
I.E., `export JULIA_NUM_THREADS=\$(nproc)`

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
    metric = Euclidean(),
    idtype::Type{I} = Int64,
    costtype::Type{D} = costtype(metric, A, B),
    executor = dynamic_thread,
    groupsize = 32,
    kw...
) where {A,B,I,D}
    runner = ExhaustiveRunner(
        length(queries),
        num_neighbors,
        I;
        executor = executor,
        costtype = D,
        max_groupsize = groupsize,
    )
    search!(runner, queries, dataset; groupsize, metric, kw...)
    return runner.groundtruth
end

const VecOrMat{T} = Union{AbstractVector{T}, AbstractMatrix{T}}
const IntOrNeighbor{I} = Union{I, <:Neighbor{I}}
inttype(::Type{I}) where {I <: Integer} = I
inttype(::Type{<:Neighbor{I}}) where {I <: Integer} = I

# """
#     ExhaustiveRunner(max_queries::Integer, num_neighbors::Integer, [ID]; [kw...])
#
# Pre-allocate data structures for at most `max_queries` with at most `num_neighbors` per query.
# Optional trailing argument `ID` selects the encoding of the results.
# It can either be an integer (defaulting to `Int64`) in which case the nearest neighbors
# will be returned as integer index, or it can be a [`Neighbor`](@ref), in which case
# index/distance pairs will be returned.
#
# ## Keyword Arguments
#
# * `executor` - Choose between [`GraphANN.single_thread`](@ref) or
#     [`GraphANN.dynamic_thread`](@ref).  Default: `GraphANN.single_thread`.
# * `costtype` - Type returned by distance computations.
# """
mutable struct ExhaustiveRunner{V <: VecOrMat{<:IntOrNeighbor}, T <: MaybeThreadLocal{<:AbstractVector}, F}
    groundtruth::V
    exhaustive_local::T
    executor::F
end

function ExhaustiveRunner(
    num_queries::Integer,
    num_neighbors::U = one,
    ::Type{ID} = Int64;
    executor::F = single_thread,
    costtype::Type{D} = Float32,
    max_groupsize = 32,
) where {ID <: IntOrNeighbor, U <: Union{Integer, typeof(one)}, F, D}
    # Preallocate destination.
    if isa(num_neighbors, Integer)
        groundtruth = Matrix{ID}(undef, num_neighbors, num_queries)
    else
        groundtruth = Vector{ID}(undef, num_queries)
    end

    # Create thread local storage.
    heaps = [KeepSmallest{Neighbor{inttype(ID),D}}(_num_neighbors(num_neighbors)) for _ in 1:max_groupsize]
    exhaustive_local = threadlocal_wrap(executor, heaps)
    return ExhaustiveRunner(groundtruth, exhaustive_local, executor)
end

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

function _Base.search!(
    runner::ExhaustiveRunner,
    queries::AbstractVector,
    dataset::AbstractVector;
    groupsize = 32,
    meter = ProgressMeter.Progress(length(queries), 1),
    metric = Euclidean(),
    skip_size_check = false,
    num_neighbors = _num_neighbors(runner.groundtruth)
)
    # Destructure runner
    @unpack groundtruth, exhaustive_local, executor = runner
    num_queries = length(queries)

    skip_size_check || sizecheck(groundtruth, num_neighbors, num_queries)

    # Batch the query range so each thread works on a chunk of queries at a time.
    # The dynamic load balancer will give one batch at a time to each worker.
    batched_iter = BatchedRange(1:num_queries, groupsize)
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
    meter !== nothing && ProgressMeter.finish!(meter)
    return groundtruth
end

#####
##### Inner functions
#####

# Define there to help out inference in the "exhaustive_search!" closure.
Base.@propagate_inbounds function _nearest_neighbors!(
    heaps::AbstractVector{KeepSmallest{T}},
    dataset::AbstractVector,
    queries::AbstractVector,
    range,
    metric,
) where {T <: Neighbor}
    @inbounds for base_id in eachindex(dataset)
        base = dataset[base_id]

        for (heap_num, query_id) in enumerate(range)
            query = queries[query_id]
            prehook(metric, query)
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

