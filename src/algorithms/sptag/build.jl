Base.@kwdef struct SPTAGBuildParams
    ### High level parameters
    target_degree::Int = 32

    ### Parameters related to tree building.
    # Number of randomized trees to help with the initial graph.
    num_trees::Int = 8
    tree_leaf_size::Int = 2000
    single_thread_threshold::Int = 10000

    ### Parameters for refinement.
    refine_iterations::Int = 2
    refine_batchsize::Int = 18000
    refine_maxcheck::Int = 8192
    refine_propagation::Int = cdiv(8192, 64)
    refine_history::Int = 2000
end

# First phase - build the initial KNN graph by partitioning.
function build_by_trees!(
    graph::UniDirectedGraph{I},
    data::AbstractVector{T};
    params = SPTAGBuildParams(),
    metric = Euclidean(),
) where {I, T}
    @unpack num_trees, tree_leaf_size, target_degree, single_thread_threshold = params
    D = costtype(metric, T)

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
    compensated_neighbors = target_degree + 1
    tls = ThreadLocal(;
        # Over-allocate the destination space in the `ExhaustiveRunner`.
        # Then, we can pass the `skip_size_check` argument to exhaustive search to only
        # populate sub sections of the result.
        runner = ExhaustiveRunner(
            Neighbor{I,D},
            tree_leaf_size,
            compensated_neighbors;
            executor = single_thread,
            costtype = D
        ),
        scratch = Neighbor{I,D}[],
    )

    for i in 1:num_trees
        GC.gc()
        @withtimer "Building TPTree" ranges = partition!(
            data,
            permutation,
            numtrials = 1000,
            Val(5);
            init = false,
            leafsize = tree_leaf_size,
            single_thread_threshold = single_thread_threshold,
        )

        neighbor_meter = ProgressMeter.Progress(length(data), 1, "Processing Tree ... ")
        @withtimer "Computing TPTree Neighbors" dynamic_thread(eachindex(ranges), 4) do i
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
            update!(
                data,
                graph,
                gt,
                view(permutation, range);
                candidates = scratch,
                metric = metric,
            )
            ProgressMeter.next!(neighbor_meter; step = length(range))
        end
    end
end

function update!(
    data,
    graph,
    gt::AbstractMatrix{T},
    permutation::AbstractVector{<:Integer};
    candidates = T[],
    metric = Euclidean(),
) where {T <: Neighbor}
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
            distance = evaluate(metric, vdata, data[u])
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

#####
##### Graph Refinement
#####

function refine!(
    graph,
    tree,
    data::AbstractVector{T};
    params::SPTAGBuildParams = SPTAGBuildParams(),
    metric = Euclidean(),
) where {T}
    @unpack target_degree, refine_history, refine_iterations, refine_batchsize = params
    D = costtype(metric, T)

    # Datastructure pre-allocation.
    tls = ThreadLocal(;
        runner = TagSearch(refine_history; costtype = D, idtype = eltype(graph)),
        nextlists = NextListBuffer{eltype(graph)}(
            target_degree,
            2 * ceil(Int, refine_batchsize / Threads.nthreads()),
        ),
    )

    # Refinement iterations.
    for i in 1:refine_iterations
        _refine!(graph, tree, data, tls; params, metric)
    end
end

@noinline function _refine!(
    graph,
    tree,
    data::AbstractVector{T},
    tls::ThreadLocal;
    params::SPTAGBuildParams = SPTAGBuildParams(),
    metric = Euclidean(),
) where {T}
    @unpack target_degree, refine_batchsize, refine_maxcheck, refine_propagation = params

    meta = MetaGraph(graph, data)
    num_batches = cdiv(length(data), refine_batchsize)
    progress_meter = ProgressMeter.Progress(num_batches, 1, "Refining Index ... ")
    for r in batched(1:length(data), refine_batchsize)
        itertime = @elapsed dynamic_thread(r, INDEX_BALANCE_FACTOR) do vertex
            storage = tls[]
            @unpack runner, nextlists = storage
            point = data[vertex]
            search(
                runner,
                meta,
                tree,
                point;
                maxcheck = refine_maxcheck,
                propagation_limit = refine_propagation,
            )

            # Obtain a nextlist to populate with new neighbors.
            nextlist = get!(nextlists)
            resize!(nextlist, target_degree)

            vertex_data = data[vertex]
            candidates = destructive_extract!(runner.results)

            count = 0
            for candidate in candidates
                getid(candidate) == vertex && continue
                good = true
                for k in 1:count
                    if evaluate(metric, data[nextlist[k]], data[candidate]) <= getdistance(candidate)
                        good = false
                        break
                    end
                end

                if good
                    count += 1
                    nextlist[count] = getid(candidate)
                end
                count == target_degree && break
            end
            resize!(nextlist, count)
            storage.nextlists[vertex] = nextlist
        end

        # Apply the refined nextlists to the graph.
        synctime = @elapsed on_threads(allthreads()) do
            @unpack nextlists = tls[]
            for (u, neighbors) in pairs(nextlists)
                copyto!(graph, u, neighbors)
            end

            # Reset nextlists for next iteration.
            empty!(nextlists)
        end

        ProgressMeter.next!(
            progress_meter;
            showvalues = ((:iter_time, itertime), (:sync_time, synctime)),
        )
    end
end

