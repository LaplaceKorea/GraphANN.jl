#####
##### Kmeans Clustering
#####

# Helper structs
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
##### Compacted representation for centroids.
#####

# Often, the number of points per centroids is much smaller than a cacheline (and thus
# AVX512 vector size.
#
# To improve computation speed, we group multiple partitions together on a single cacheline
# and compute multiple distances in parallel.
struct PackedCentroids{K,E,V}
    # Centroids stored in a transposed form.
    centroids::Matrix{Packed{K,E,V}}
    # Is the corresponding entry in `centroids` valid?
    masks::Matrix{SIMD.Vec{K,Bool}}
    # How many centroids are selected for each partition.
    lengths::Vector{Int}
end

function PackedCentroids{E,V}(
    num_partitions::Integer,
    centroids_per_partition::Integer
) where {E <: Euclidean, V <: SIMD.Vec}
    # This is pretty hacky at the moment, but the idea is to first allocate initialize
    # the space we need, then reinterpret and collect.
    # Since the amount of memory occupied by the centroids should be pretty low, we don't
    # have much overhead in unnecessarily copying this data around.
    K = exact_div(length(V), length(E))
    num_groups = exact_div(num_partitions, K)
    raw = zeros(V, num_groups, centroids_per_partition)
    centroids = raw |>
        x -> reinterpret(Packed{K, E, V}, x) |>
        collect

    masks = zeros(SIMD.Vec{K,Bool}, size(centroids))
    lengths = zeros(Int, num_partitions)
    return PackedCentroids{K,E,V}(centroids, masks, lengths)
end

Base.size(x::PackedCentroids) = size(x.centroids)
Base.size(x::PackedCentroids, i::Integer) = size(x.centroids, i)
fullsize(x::PackedCentroids{K}) where {K} = (K * size(x, 1), size(x, 2))

Base.eltype(x::PackedCentroids{K,E,V}) where {K,E,V} = Packed{K,E,V}
Base.getindex(x::PackedCentroids, i...) = getindex(x.centroids, i...)
Base.get(x::PackedCentroids, i...) = get(x.centroids, i...)
getmask(x::PackedCentroids, i...) = getindex(x.masks, i...)
maxlength(x::PackedCentroids) = maximum(x.lengths)
lengthof(x::PackedCentroids, i::Integer) = getindex(x.lengths, i)

function Base.push!(x::PackedCentroids{K,E}, v::E, offset, group) where {K,E}
    @assert 1 <= offset <= K
    @unpack centroids, masks, lengths = x
    # Insert this data point in the correct location
    partition = (group - 1) * K + offset
    horizontal_index = lengths[partition] + 1
    horizontal_index > size(x, 2) && return nothing

    # Insert new data point
    _Points.set!(centroids, v, offset, group, horizontal_index)

    # Update masks to indicate that this slot is filled.
    # Need to do a little dance with constructing a `SIMD.Vec` from a tuple.
    new_bit = SIMD.Vec(ntuple(i -> (i == offset), Val(K)))
    masks[group, horizontal_index] |= new_bit
    lengths[partition] = horizontal_index
    return nothing
end

#####
##### Entry Point
#####

function choose_centroids(
    centroids_per_partition::Integer,
    data::Vector{Euclidean{N,T}},
    num_centroids::Integer;
    num_iterations = 2,
    oversample = 1.3,
) where {T,N,S,U}
    # What is the centroid type
    centroid_type = Euclidean{centroids_per_partition,T}
    packed_type = _Points.packed_type(centroid_type)
    num_partitions = exact_div(N, length(centroid_type))

    # Wrap the data in the correct view type
    data_wrapped = LazyArrayWrap{packed_type}(data)

    # Create a packed representation for the centroids
    centroids = PackedCentroids{centroid_type, packed_type}(
        num_partitions,
        ceil(Int, num_centroids * oversample)
    )

    # Inner function to handle any type instability at this upper level.
    choose_centroids!(
        centroids,
        data_wrapped,
        num_iterations,
    )

    return centroids
end

function choose_centroids!(
    packed_centroids::PackedCentroids{K,E,V},
    data::LazyArrayWrap,
    num_iterations::Integer,
) where {K,E,V}
    @show Tuple{typeof(packed_centroids), typeof(data), typeof(num_iterations)}

    # Hoist up a bunch of useful constants
    # -- types
    P = Packed{K,E,V}

    # -- values
    num_centroids = size(packed_centroids, 2)
    num_groups = size(packed_centroids, 1)
    offsets = 1:K
    num_partitions = num_groups * K
    partitions = 1:num_partitions
    data_as_partitions = LazyArrayWrap{E}(parent(data))
    samples_per_iteration = ceil(Int, num_centroids / num_iterations)

    # Create a cost per centroid.
    #costs = Matrix{Int32}(undef, num_partitions, size(data, 2))
    costs = zeros(Int32, num_partitions, size(data, 2))
    total_costs = Vector{Int64}(undef, num_partitions)
    tls = ThreadLocal(;
        current_minimums = [CurrentMinimum{K,Int32}() for _ in 1:num_groups],
        local_centroids = [Set{E}() for _ in 1:num_partitions],
    )

    # Initialize centroids randomly
    for group in 1:num_groups, offset in 1:K
        # Since `data` is wrapped in a wider type, we need to use `LazyWrap` to
        # convert to the actual centroid type.
        partition = K * (group - 1) + offset
        centroid = rand(view(data_as_partitions, partition, :))
        push!(packed_centroids, centroid, offset, group)
    end

    for iter in 1:num_iterations
        # Step 1 - Compute the costs for all data points.
        maxind = maxlength(packed_centroids)
        dynamic_thread(1:size(data, 2), 1024) do i
            @unpack current_minimums = tls[]
            @unpack centroids, masks = packed_centroids

            # Find the closest centroids.
            findcenters!(
                current_minimums,
                centroids,
                view(data, :, i),
                maxind,
                masks,
            )

            # Update the costs matrix
            updatecosts!(view(costs, :, i), current_minimums)
        end

        # Step 2 - total up the costs per centroid.
        sumcosts!(total_costs, costs)
        @show sum(total_costs)

        # Step 3 - Choose more centroids.
        dynamic_thread(1:size(data, 2), 1024) do i
        #for i in 1:size(data, 2)
            @unpack local_centroids = tls[]
            @inbounds process_partition(
                local_centroids,
                view(data_as_partitions, :, i),
                view(costs, :, i),
                total_costs,
                samples_per_iteration,
            )
        end

        # Step 4 - Union all centroids and update
        for thread_local in getall(tls)
            for group in 1:num_groups, offset in offsets
                partition = K * (group - 1) + offset
                new_centroids = thread_local.local_centroids[partition]
                for new_centroid in new_centroids
                    push!(packed_centroids, new_centroid, offset, group)
                end
                empty!(new_centroids)
            end
        end
    end
    return packed_centroids
end

function findcenters!(
    current_minimums::AbstractVector{C},
    centroids_transposed::AbstractMatrix{P},
    data::AbstractVector,
    maxind = size(centroids_transposed, 2),
    masks = nothing,
) where {C <: CurrentMinimum, P <: Packed}
    # Do we have the correct number of partitions?
    @assert size(data, 1) == size(centroids_transposed, 1)

    # Reset minimums
    current_minimums .= C()
    for j in 1:maxind
        @inbounds for i in 1:size(centroids_transposed, 1)
            # Wrap this partition of the datapoint in a `Packed` representation.
            centroids = centroids_transposed[i, j]
            d = distance(centroids, P(data[i]))
            # TODO - this is a hack fow now ...
            # If a mask set is provided, than only grab the computed distances for which
            # the distance is set.
            if masks !== nothing
                d = SIMD.vifelse(masks[i, j], d, current_minimums[i].distance)
            end
            current_minimums[i] = update(current_minimums[i], d, j)
        end
    end
end

function updatecosts!(
    costs::AbstractVector,
    current_minimums::AbstractVector{C},
) where {K, C <: CurrentMinimum{K}}
    @assert length(costs) == K * length(current_minimums)
    for (i, minimums) in enumerate(current_minimums)
        for (j, cost) in enumerate(Tuple(minimums.distance))
            costs[K * (i-1) + j] = cost
        end
    end
end


function sumcosts!(
    total_costs::AbstractVector{T},
    individual_costs::AbstractMatrix
) where {T}
    # Size check
    if length(total_costs) != size(individual_costs, 1)
        throw(ArgumentError("""
        The length of argument `total_costs` must be the same the `size(individual_costs, 1)`.
        Instead, they are $(length(total_costs)) and $(size(individual_costs, 1)) respectively.
        """))
    end
    zero!(total_costs)
    tls = ThreadLocal(zeros(T, length(total_costs)))

    # Locally accumulate per thread.
    dynamic_thread(1:size(individual_costs, 2), 1024) do j
        local_costs = tls[]
        @inbounds for i in eachindex(local_costs)
            local_costs[i] += convert(T, individual_costs[i, j])
        end
    end

    # Aggregate across threads.
    total_costs .= sum(getall(tls))
    return total_costs
end

function process_partition(
    centroids,
    samples::AbstractVector,
    costs::AbstractVector,
    total_costs::AbstractVector,
    adjustment::Number,
)
    for i in eachindex(samples)
        sample_probability = adjustment * costs[i] / total_costs[i]
        if rand() < sample_probability
            push!(centroids[i], samples[i])
        end
    end
end

# Refinement - essentially kmeans++
function refine(x::PackedCentroids{K,E,V}, y::LazyArrayWrap{V}, args...) where {K,E,V}
    return refine(x, LazyArrayWrap{E}(parent(y)), args...)
end

function refine(x::PackedCentroids{K,E}, y::AbstractVector, args...) where {K,E}
    return refine(x, LazyArrayWrap{E}(y), args...)
end

function refine(
    packed_centroids::PackedCentroids{K,E,V},
    data::LazyArrayWrap{E},
    num_centroids::Integer
) where {K,E,V}
    data_as_partitions = LazyArrayWrap{E}(parent(data))
    final_centroids = Array{E}(undef, num_centroids, size(data,1))
    unpacked_centroids = reinterpret(E, packed_centroids.centroids)

    # First - determine how many points are assigned to each centroid.
    weights = counts_per_centroid(packed_centroids, data)
    dynamic_thread(1:size(data, 1)) do partition
        centroids_in_partition = lengthof(packed_centroids, partition)

        # Construct a bunch of views to pass to the final `refine` function.
        # Since this function is fairly light weight, we don't spend a whole lot of time
        # optimizing it.
        refine!(
            view(final_centroids, :, partition),
            view(unpacked_centroids, partition, 1:centroids_in_partition),
            view(weights, partition, 1:centroids_in_partition),
        )
    end
    return final_centroids
end

function counts_per_centroid(
    packed_centroids::PackedCentroids{K,E,V},
    data::LazyArrayWrap{E}
) where {K,E,V}
    return counts_per_centroid(packed_centroids, LazyArrayWrap{V}(parent(data)))
end

function counts_per_centroid(
    packed_centroids::PackedCentroids{K,E,V},
    data::LazyArrayWrap{V},
) where {K,E,V}
    num_groups = size(packed_centroids, 1)
    tls = ThreadLocal(;
        minimums = [CurrentMinimum{K,Int32}() for _ in 1:num_groups],
        counts = zeros(Int, fullsize(packed_centroids))
    )

    maxind = maxlength(packed_centroids)
    dynamic_thread(1:size(data, 2), 1024) do i
        @unpack minimums, counts = tls[]
        @unpack centroids, masks = packed_centroids
        findcenters!(minimums, centroids, view(data, :, i), maxind, masks)

        # Accumulate counts
        for (group, mins) in enumerate(minimums)
            for (offset, index) in enumerate(Tuple(mins.index))
                # If this is a valid entry, than accumulate the counts.
                if masks[group, index][offset] == true
                    partition = K * (group - 1) + offset
                    counts[partition, index] += 1
                end
            end
        end
    end

    # Accumulate results across all threads.
    return sum(x -> x.counts, getall(tls))
end

function refine!(
    final_centroids::AbstractVector{E},
    centroids::AbstractVector{E},
    weights::AbstractVector,
) where {N, T, E <: Euclidean{N,T}}
    costs = Vector{Float32}(undef, length(centroids))
    num_centroids = length(final_centroids)

    # Select a random centroid to begin with.
    centroids_found = 1
    final_centroids[centroids_found] = rand(centroids)
    while centroids_found < num_centroids
        # Compute minimum costs
        for i in eachindex(centroids)
            costs[i] = minimum(
                x -> weights[i] * distance(centroids[i], x),
                view(final_centroids, 1:centroids_found),
            )
        end
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
        while index <= length(centroids) && accumulator < threshold
            index += 1
            accumulator += costs[index] / total_cost
        end

        centroids_found += 1
        final_centroids[centroids_found] = centroids[index]
    end
    return nothing
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
    # Recast the data array along partition boundaries to facilitate selecting
    # new centroids if needed.
    data_as_partitions = LazyArrayWrap{E}(parent(data))

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
        # 3. Repick centroids that have no assignments.
        integrated_points = mapreduce(x -> x.integrators, (x,y) -> x .+ y, getall(tls))
        points_per_center = mapreduce(x -> x.points_per_center, +, getall(tls))

        for i in CartesianIndices(integrated_points)
            ppc = points_per_center[i]
            # If no points are assigned to this centroid, then repick the centroid.
            if iszero(ppc)
                offset, group, centroid_number = Tuple(i)
                partition = K * (group - 1) + offset
                data_view = view(data_as_partitions, partition, :)
                set!(centroids, rand(data_view), i)
            # Otherwise, shift the centroid to be the median of all the points assigned
            # to it.
            else
                new_centroid = integrated_points[i] / ppc
                set!(centroids, new_centroid, i)
            end
        end

        # Reset for next run.
        foreach(x -> zero!(x.integrators), getall(tls))
        foreach(x -> zero!(x.points_per_center), getall(tls))
    end
    return tls
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

