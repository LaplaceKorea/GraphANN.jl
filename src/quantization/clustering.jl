#####
##### Kmeans Clustering
#####

# Compute the individual distances squared between a dataset and a collection of centroids.
function compute_cost!(
    costs::AbstractVector{T},
    data,
    centroids;
    worksize = 1024
) where {T}
    # Here, the worksize i
    dynamic_thread(batched(eachindex(data), worksize)) do range
        # Initialize this range of costs.
        for i in range
            costs[i] = typemax(eltype(costs))
        end

       # Compute minimum in a batched manner.
        for centroid in centroids, i in range
            @inbounds datum = data[i]
            @inbounds costs[i] = min(costs[i], distance(datum, centroid))
        end
    end
    return costs
end

# This is basically an implementation of kmeans++, means to run on the reduced number
# of clusters computed by `choose_centroids`.
function refine(data::AbstractVector, num_centroids)
    # Choose the first centroid at random.
    costs = Vector{Int32}(undef, length(data))
    centroids = Vector{eltype(data)}()
    push!(centroids, data[rand(1:length(data))])

    while length(centroids) < num_centroids
        # Choose the next centroid based on the its distance from the current set of
        # centroids.
        compute_cost!(costs, data, centroids)
        total_cost = sum(costs)

        # The strategy we use is to generate a random number uniformly between 0 and 1.
        # Then, we move through the `costs` vector, accumulating results as we go until
        # we find the first point where the accumulated value is greater than the
        # threshold.
        #
        # This is the point we choose.
        threshold = rand()
        accumulator = zero(threshold)
        index = 0
        while index <= length(data) && accumulator < threshold
            index += 1
            accumulator += costs[index] / total_cost
        end
        push!(centroids, data[index])
    end
    return centroids
end

#####
##### Lloyd's algorithm
#####

struct CurrentMinimum{N,T}
    distance::SIMD.Vec{N,T}
    index::SIMD.Vec{N,Int32}
end

function CurrentMinimum{N,T}() where {N,T}
    return CurrentMinimum{N,T}(SIMD.Vec{N,T}(typemax(T)), zero(SIMD.Vec{N,Int32}))
end
# Scalar broadcasting
Base.broadcastable(x::CurrentMinimum) = (x,)
Base.isless(a::CurrentMinimum, b::CurrentMinimum) = (a.distance < b.distance)
Base.length(x::CurrentMinimum{N}) where {N} = N

function update(x::CurrentMinimum{N,T}, d::SIMD.Vec{N,T}, i::Integer) where {N,T}
    mask = d < x.distance
    index_update = SIMD.Vec{N,Int32}(Int32(i))

    distance = SIMD.vifelse(mask, d, x.distance)
    index = SIMD.vifelse(mask, index_update, x.index)
    return CurrentMinimum{N,T}(distance, index)
end

#####
##### Round 2 - FIGHT!
#####

# Do the clustering in parallel because it's REALLY slow to do one partition at a time.
# Choose initial centroids based on k-means++
# https://en.wikipedia.org/wiki/K-means
# http://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf
function choose_centroids(
    data::LazyArrayWrap{Euclidean{N,T}, S, U},
    num_centroids::Integer;
    num_iterations = 2,
    oversample = 1.3,
) where {T,N,S,U}
    num_partitions = div(S,N)
    partitions = 1:num_partitions

    # One cost per data point for each partition.
    # Use Float32 to save on space.
    costs = Matrix{Int32}(undef, size(data))

    # Centroids chosend for each partiton.
    # Make the centroid have the same type as that yielded by the LazyWrap.
    centroids_sets = [Set{Euclidean{N,T}}() for _ in partitions]

    # Choose the initial centroid uniformly at random from the dataset
    for (partition, centroids) in enumerate(centroids_sets)
        # Select a random entry in the data and pull out the corresponding
        push!(centroids, rand(view(data, partition, :)))
    end

    compute_cost!(costs, data, centroids_sets)
    total_cost = map(row -> sum(x -> convert(Int64, x), row), eachrow(costs))
    samples_per_iteration = ceil(Int, num_centroids * oversample / num_iterations)

    # Display updates
    printstyled("Number of Iterations: "; color = :green, bold = :true)
    println(num_iterations)

    # Keep local centroids for each partition
    local_centroids = ThreadLocal([Set{Euclidean{N,T}}() for _ in 1:num_partitions])
    #ProgressMeter.@showprogress 1 for _iter in 1:num_iterations
    for _iter in 1:num_iterations
        # Independent sampling of the dataset
        # Need to pull the `let` trick because Julia's struggling with the capturing
        # of `total_cost` in the closure below.
        let total_cost = total_cost
            dynamic_thread(1:size(data, 2), 1024) do i
                for partition in partitions
                    # Sample this datum proportionally with its current cost
                    partition_total = total_cost[partition]
                    sample_probability = samples_per_iteration * costs[partition, i] / partition_total
                    if rand() < sample_probability
                        push!(local_centroids[][partition], data[partition, i])
                    end
                end
            end
        end

        # Update current list of centroids
        for local_centroid_sets in getall(local_centroids)
            for (centroids, local_centroids) in zip(centroids_sets, local_centroid_sets)
                union!(centroids, local_centroids)
            end
        end

        # Recompute total cost
        compute_cost!(costs, data, centroids_sets)
    end

    # Choose the final candidates from this collection.
    printstyled("Finished initial run with "; color = :green, bold = true)
    print(length.(centroids_sets))
    printlnstyled(" centroids."; color = :green, bold = true)

    refined = map(centroids_sets) do centroids
        refine(collect(centroids), num_centroids)
    end
    return reduce(hcat, refined)
end

function compute_cost!(
    costs::AbstractMatrix,
    data::LazyArrayWrap{Euclidean{N,T}, S, U},
    centroid_sets::AbstractVector{<:AbstractSet};
    worksize = 1024,
) where {N,T,S,U}
    num_partitions = div(S, N)
    partitions = 1:num_partitions

    dynamic_thread(batched(1:size(data, 2), worksize)) do range
        # Initialize costs
        typemax!(view(costs, partitions, range))

        for (partition, centroids) in enumerate(centroid_sets)
            for centroid in centroids, i in range
                @inbounds datum = data[partition, i]
                @inbounds costs[partition, i] = min(
                    costs[partition, i],
                    distance(datum, centroid),
                )
            end
        end
    end
    return costs
end


#####
##### Batched `lloyds` algorithm.
#####

# Step 1 - Convert centroids to be Float32
function lloyds(centroids::AbstractMatrix{Euclidean{N,T}}, x...; kw...) where {N,T}
    return lloyds(map(i -> convert(Euclidean{N,Float32}, i), centroids), x...; kw...)
end

# Step 2 - Transpose and Pack Centroids
function lloyds(
    centroids::AbstractMatrix{Euclidean{N,Float32}},
    data::AbstractVector{<:Euclidean};
    kw...
) where {N}
    # Compute the number of centroids to pack per cacheline
    K = div(16, N)
    packed_centroids = centroids |>
        transpose |>
        x -> reinterpret(Packed{K, Euclidean{N,Float32}, SIMD.Vec{16,Float32}}, x) |>
        collect

    wrapped_data = LazyArrayWrap{SIMD.Vec{16,Float32}}(data)
    lloyds!(packed_centroids, wrapped_data; kw...)

    # Unwrap the centroids and return.
    return packed_centroids |>
        transpose |>
        x -> reinterpret(Euclidean{N,Float32}, x) |>
        x -> reshape(x, size(centroids)) |>
        collect
end

# Step 3 - actually run the algorithm
# The idea here is that we compute distances between a query vector and a whole set of
# centroid partitions at a time.
#
# To do this, we need to convert the layout of the `centroids` matrix to be as follows.
# Let `Cᵢⱼ` represent the data points for a the ith centroid of partition `j`.
# We can group centroids from subsequent partitions together on a cache line to take
# advantage of extra SIMD space.
#
#   Query:      Q₁  Q₂  … Qₙ
#               --------------
# Centroids:    C₁₁ C₁₂ … C₁ₙ       <--- Memory Order --->
#               C₂₁ C₂₂ … C₂ₙ
#               C₃₁ C₃₂ … C₃ₙ
#                 ⋮       ⋮
#               Cₚ₁ Cₚ₂ … Cₚₙ
#
#
function lloyds!(
    centroids::AbstractMatrix{<:Packed{K,E}},
    data::LazyArrayWrap;
    num_iterations = 1,
) where {K,E}
    num_groups = size(centroids, 1)
    num_centroids = size(centroids, 2)
    num_partitions = K * num_groups

    # Allocate thread local storage
    tls = ThreadLocal(;
        current_minimums = [CurrentMinimum{K,Float32}() for _ in 1:num_groups],
        integrators = zeros(E, K, num_groups, num_centroids),
        points_per_center = zeros(Int, K, num_groups, num_centroids),
    )

    for iter in 1:num_iterations
        @time dynamic_thread(1:size(data, 2), 1024) do i
            threadlocal = tls[]
            v = view(data, :, i)
            findcenters!(threadlocal.current_minimums, centroids, v)
            updatelocal!(E, threadlocal, v)
        end

        # Now that we've accumulated everything on each thread, we have several things to do.
        # 1. Accumulate integrator values and points-per-center for each centroid.
        # 2. Update centroids to be at the center of their region.
        # 3. (TODO) repick centroids that have no assignments.
        integrated_points = mapreduce(x -> x.integrators, (x,y) -> x .+ y, getall(tls))
        points_per_center = mapreduce(x -> x.points_per_center, +, getall(tls))

        for i in CartesianIndices(integrated_points)
            ppc = points_per_center[i]
            if !iszero(ppc)
                new_centroid = integrated_points[i] / ppc
                #old_centroid = get(centroids, i)
                #@show sum(new_centroid - old_centroid)
                set!(centroids, new_centroid, i)
            end
        end

        # Reset for next run.
        foreach(x -> zero!(x.integrators), getall(tls))
        foreach(x -> zero!(x.points_per_center), getall(tls))
    end
    return tls
end

function findcenters!(
    current_minimums::AbstractVector{C},
    centroids_transposed::AbstractMatrix{P},
    data::AbstractVector,
) where {C <: CurrentMinimum, P <: Packed}
    #data = view(data, :, col)
    # Do we have the correct number of partitions?
    #@assert size(data, 1) == size(centroids_transposed, 1)
    num_partitions = size(centroids_transposed, 1)

    # Reset minimums
    current_minimums .= C()
    for j in 1:size(centroids_transposed, 2)
        @inbounds for i in 1:size(centroids_transposed, 1)
            # Wrap this partition of the datapoint in a `Packed` representation.
            centroids = centroids_transposed[i, j]
            d = distance(centroids, P(data[i]))
            current_minimums[i] = update(current_minimums[i], d, j)
        end
    end
end

function updatelocal!(
    ::Type{E},
    threadlocal::NamedTuple,
    data::AbstractVector,
) where {E}
    @unpack current_minimums, integrators, points_per_center = threadlocal
    for (group, minimums) in enumerate(current_minimums)
        slice = LazyWrap{E}(data[group])
        assignments = minimums.index

        for subgroup in 1:length(minimums)
            assignment = assignments[subgroup]
            integrators[subgroup, group, assignment] += slice[subgroup]
            points_per_center[subgroup, group, assignment] += 1
        end
    end
end

