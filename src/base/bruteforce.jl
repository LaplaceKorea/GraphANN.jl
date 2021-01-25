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
    queries::AbstractVector{A},
    dataset::AbstractVector{B},
    num_neighbors::Int = 100;
    groupsize = 32,
    savefile = nothing,
    idtype::Type{T} = UInt32,
    metric::F = distance,
) where {A,B,T,F}
    # Allocate max heaps for each
    # One for each column in the queries matrix
    D = costtype(A, B)
    _heaps = [BoundedMaxHeap{Neighbor{T,D}}(num_neighbors) for _ in 1:groupsize]
    tls = ThreadLocal(_heaps)

    num_queries = length(queries)
    meter = ProgressMeter.Progress(num_queries, 1)

    # Pre-allocate ground-truth array
    # TODO: Parameterize integer type?
    gt = Array{T,2}(undef, num_neighbors, num_queries)

    # Batch the query range so each thread works on a chunk of queries at a time.
    # The dynamic load balancer will give one batch at a time to each worker.
    batched_iter = BatchedRange(1:num_queries, groupsize)
    dynamic_thread(batched_iter) do range
        heaps = tls[]

        # Compute nearest neighbors for this batch across the whole dataset.
        for (base_id, base) in enumerate(dataset)
            for (heap_num, query_id) in enumerate(range)
                query = queries[query_id]
                dist = metric(query, base)

                # Need to convert from 1 based indexing to 0 based indexing...
                push!(heaps[heap_num], Neighbor{T}(base_id - 1, dist))
            end
        end

        # Commit results
        for (heap_num, query_id) in enumerate(range)
            vgt = view(gt, :, query_id)
            vgt .= getid.(DataStructures.extract_all_rev!(heaps[heap_num].heap))
        end

        # Note: ProgressMeter is threadsafe - so calling it here is okay.
        # With modestly sized dataset, the arrival time of threads to this point should be
        # staggered enough that lock contention shouldn't be an issue.
        #
        # For reference, it takes ~200 seconds to compute 100 nearest neighbors for a
        # batch of 32 queries on Sift100M.
        #
        # That is FAR longer than any time will be spent fighting over this lock.
        ProgressMeter.next!(meter; step = length(range))

        if savefile !== nothing && Threads.threadid() == 1
            # NOTE: small race condition with updating `gt` while this is tryint to
            # read it.
            #
            # However, it shouldn't affect the actual saved data.
            #
            # Do the trick of serializing to a temporary file and then moving to always
            # ensure the saved file is valid.
            tempfile = "$savefile.temp"
            save_vecs(tempfile, gt)
            mv(tempfile, savefile; force = true)
        end
    end
    ProgressMeter.finish!(meter)

    # final save
    savefile === nothing || save_vecs(savefile, gt)

    return gt
end
