# First phase - build the initial KNN graph by partitioning.
function build_by_trees!(
    graph::UniDirectedGraph,
    data::AbstractVector{T};
    num_iterations = 1,
    num_neighbors = 32,
    leafsize = 500,
    idtype::Type{I} = UInt32,
    costtype::Type{D} = costtype(T, T),
    executor = dynamic_thread
) where {T, I, D}
    # Preallocate utility for accelerating the partitioning phase
    tree = TPTree{5,I}(length(data))

    # Make the leaf callback simply queue up ranges.
    ranges = Vector{UnitRange{I}}()
    groupsize = 32

    # Pre-allocate destination matrices for the brute-force nearest neighbor searches.
    # Since the size of the nearest neighbor searches is bounded by the leaf size, we
    # can pre-allocate for the worst case and use views to make these arrays as small
    # as needed.
    #
    # Also, keep the element type of the ground-truth buffers as `Neighbors` to more
    # efficiently compare with existing neighbors in the graph.
    tls = ThreadLocal(;
        groundtruth = Array{Neighbor{I,D}, 2}(undef, num_neighbors, leafsize),
        bruteforce_tls = _Base.bruteforce_threadlocal(single_thread, I, D, num_neighbors, groupsize)
    )

    for i in 1:num_iterations
        empty!(ranges)
        progress_meter = ProgressMeter.Progress(length(data), 1, "Building Tree ... ")
        @time partition!(data, tree; init = false) do leaf
            ProgressMeter.next!(progress_meter; step = length(leaf))
            push!(ranges, leaf)
        end

        ProgressMeter.finish!(progress_meter)
        progress_meter = ProgressMeter.Progress(length(data), 1, "Processing Tree ... ")
        executor(eachindex(ranges), 16) do i
            @inbounds range = ranges[i]
            threadlocal = tls[]

            gt = view(threadlocal.groundtruth, :, 1:length(range))
            dataview = viewdata(data, tree, range)
            bruteforce_search!(gt, dataview, dataview;
                executor = single_thread,
                idtype = I,
                costtype = costtype,
                meter = nothing,
                tls = threadlocal.bruteforce_tls,
                groupsize = groupsize,
            )

            # Update graph to maintain nearest neighbors.
            update!(data, graph, gt, viewperm(tree, range))

            ProgressMeter.next!(progress_meter; step = length(range))
        end
    end
end

function update!(data, graph, gt::AbstractMatrix{T}, permutation::AbstractVector{<:Integer}) where {T <: Neighbor}
    # First step - translate the indices in `gt` from the local versions to the global
    # vertices using the permutation vector.
    for i in eachindex(gt)
        # Add 1 to the ID to convert from index-0 to index-1.
        neighbor = gt[i]
        gt[i] = T(permutation[getid(neighbor) + 1], getdistance(neighbor))
    end

    # Now that the global ID's have been computed, we need to extract the nearest neighbors
    # between the current neighbors and the new candidates.
    candidates = T[]
    numneighbors = size(gt, 1) - 1
    for (i, v) in enumerate(permutation)
        neighbors = LightGraphs.outneighbors(graph, v)
        vdata = data[v]

        # Drop the first entry because that will always be zero since each vertex is its
        # own absolute nearest neighbor.
        resize!(candidates, numneighbors)
        candidates .= view(gt, 2:size(gt,1), i)

        for u in neighbors
            distance = evaluate(Euclidean(), vdata, data[u])
            neighbor = T(u, distance)
            if !in(neighbor, candidates)
                push!(candidates, neighbor)
            end
        end
        sort!(candidates)
        empty!(graph, v)
        for i in 1:numneighbors
            LightGraphs.add_edge!(graph, v, getid(candidates[i]))
        end
    end
end
