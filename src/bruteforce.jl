#####
##### Bounded Heap
#####

struct BruteForceHeap{H}
    heap::H
    bound::Int
end

function BruteForceHeap{T}(bound::Int) where {T}
    return BruteForceHeap(
        DataStructures.BinaryMaxHeap{T}(),
        bound,
    )
end

function Base.push!(H::BruteForceHeap, i)
    if (length(H.heap) < H.bound || i < first(H.heap))
        push!(H.heap, i)
        if length(H.heap) > H.bound
            pop!(H.heap)
        end
    end
    return nothing
end

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
    queries::AbstractVector{T},
    dataset::AbstractVector{T},
    num_neighbors::Int = 100;
    groupsize = 32,
) where {T}
    # Allocate max heaps for each
    # One for each column in the queries matrix
    _heaps = [BruteForceHeap{Neighbor}(num_neighbors) for _ in 1:groupsize]
    tls = ThreadLocal(_heaps)

    # Not the most efficient way of doing this, but whatever
    # Note: ProgressMeter is threadsafe
    num_queries = length(queries)
    meter = ProgressMeter.Progress(num_queries, 1)

    gt = Array{UInt64,2}(undef, num_neighbors, num_queries)

    batched_iter = BatchedRange(1:num_queries, groupsize)
    dynamic_thread(batched_iter) do range
        heaps = tls[]
        for (base_id, base) in enumerate(dataset)
            for (heap_num, query_id) in enumerate(range)
                query = queries[query_id]
                dist = distance(query, base)

                # Need to convert from 1 based indexing to 0 based indexing...
                push!(heaps[heap_num], Neighbor(base_id - 1, dist))
            end
        end

        # Commit results
        for (heap_num, query_id) in enumerate(range)
            vgt = view(gt, :, query_id)
            vgt .= getid.(DataStructures.extract_all_rev!(heaps[heap_num].heap))
        end

        ProgressMeter.next!(meter; step = length(range))
    end
    ProgressMeter.finish!(meter)

    return gt
end
