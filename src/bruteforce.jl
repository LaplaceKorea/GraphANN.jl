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

function bruteforce_search(
    queries::AbstractVector{T},
    dataset::AbstractVector{T},
    num_neighbors::Int = 100
) where {T}
    # Allocate max heaps for each
    # One for each column in the queries matrix
    heaps = [BruteForceHeap{Neighbor}(num_neighbors) for _ in 1:size(queries,2)]

    # Not the most efficient way of doing this, but whatever
    meter = ProgressMeter.Progress(
        div(size(queries, 2), Threads.nthreads()),
        1,
    )

    Threads.@threads for query_id in 1:size(queries, 2)
        query = queries[query_id]
        for (base_id, base) in enumerate(dataset)
            dist = distance(query, base)
            heap = heaps[query_id]

            # Need to convert from 1 based indexing to 0 based indexing...
            push!(heap, Neighbor(base_id - 1, dist))
        end

        if Threads.threadid() == 1
            ProgressMeter.next!(meter)
        end
    end

    # Extract all results into a single 2d array
    dest = Array{Int32,2}(undef, num_neighbors, length(heaps))
    for (i, col) in enumerate(eachcol(dest))
        col .= getid.(DataStructures.extract_all_rev!(heaps[i].heap))
    end

    return dest
end
