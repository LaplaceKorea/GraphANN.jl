# First phase - build the initial KNN graph by partitioning.
function build_by_trees!(
    graph::UniDirectedGraph{I},
    data::AbstractVector{T};
    num_iterations = 1,
    num_neighbors = 32,
    leafsize = 500,
    single_thread_threshold = 10000,
) where {I, T}
    D = costtype(T, T)

    # Preallocate utility for accelerating the partitioning phase
    permutation = collect(I(1):I(lastindex(data)))

    # Make the leaf callback simply queue up ranges.  ranges = Vector{UnitRange{I}}()
    # Pre-allocate destination matrices for the brute-force nearest neighbor searches.
    # Since the size of the nearest neighbor searches is bounded by the leaf size, we
    # can pre-allocate for the worst case and use views to make these arrays as small
    # as needed.
    #
    # Also, keep the element type of the ground-truth buffers as `Neighbors` to more
    # efficiently compare with existing neighbors in the graph.
    #
    # Add 1 to the `num_neighbors` when constructing the thread local groundtruth buffer
    # since eacy vertex will appear as its own nearest neighbor, effectively removing
    # it from consideration.
    compensated_neighbors = num_neighbors + 1
    tls = ThreadLocal(;
        # Over-allocate the destination space in the `ExhaustiveRunner`.
        # Then, we can pass the `skip_size_check` argument to exhaustive search to only
        # populate sub sections of the result.
        runner = ExhaustiveRunner(
            Neighbor{I,D},
            leafsize,
            compensated_neighbors;
            executor = single_thread,
            costtype = D
        ),
        scratch = Neighbor{I,D}[],
    )

    for i in 1:num_iterations
        ranges = partition!(
            data,
            permutation,
            Val(5);
            init = false,
            leafsize = leafsize,
            single_thread_threshold = single_thread_threshold,
        )

        neighbor_meter = ProgressMeter.Progress(length(data), 1, "Processing Tree ... ")
        dynamic_thread(eachindex(ranges), 16) do i
            @inbounds range = ranges[i]
            @unpack runner, scratch = tls[]

            ub = min(compensated_neighbors, length(range))
            dataview = doubleview(data, permutation, range)
            groundtruth = search!(
                runner,
                dataview,
                dataview;
                meter = nothing,
                num_neighbors = ub,
                skip_size_check = true,
            )
            gt = view(groundtruth, 1:ub, 1:length(range))

            # Update graph to maintain nearest neighbors.
            update!(data, graph, gt, view(permutation, range); candidates = scratch)
            ProgressMeter.next!(neighbor_meter; step = length(range))
        end
    end
end

function update!(data, graph, gt::AbstractMatrix{T}, permutation::AbstractVector{<:Integer}; candidates = T[]) where {T <: Neighbor}
    # First step - translate the indices in `gt` from the local versions to the global
    # vertices using the permutation vector.
    for i in eachindex(gt)
        # Add 1 to the ID to convert from index-0 to index-1.
        neighbor = gt[i]
        gt[i] = T(permutation[getid(neighbor) + 1], getdistance(neighbor))
    end

    # Now that the global ID's have been computed, we need to extract the nearest neighbors
    # between the current neighbors and the new candidates.
    empty!(candidates)
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
        sort!(candidates; alg = Base.InsertionSort)
        empty!(graph, v)
        for i in 1:numneighbors
            LightGraphs.add_edge!(graph, v, getid(candidates[i]))
        end
    end
end
