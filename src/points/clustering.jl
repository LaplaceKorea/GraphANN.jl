#####
##### Choose Initial Points
#####

# Choose initial centroids based on k-means++
# https://en.wikipedia.org/wiki/K-means
# http://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf
function choose_centroids(
    data::AbstractVector{T},
    num_centroids::Integer;
    num_iterations = 10,
    oversample = 5,
) where {T}
    # Use Float64 for the costs for better accuracy.
    # May be able to use Float32 just fine though.
    costs = Vector{Float64}(undef, length(data))
    centroids = Set{T}()

    # Choose the initial centroid uniformly at random from the dataset
    push!(centroids, data[rand(1:length(data))])
    compute_cost!(costs, data, centroids)
    total_cost = sum(costs)
    samples_per_iteration = ceil(Int, num_centroids * oversample / num_iterations)

    # Display updates
    printstyled("Initial Cost: "; color = :green, bold = :true)
    println(total_cost)
    printstyled("Number of Iterations: "; color = :green, bold = :true)
    println(num_iterations)

    local_centroids = ThreadLocal(Set{T}())
    ProgressMeter.@showprogress 1 for _ in 1:num_iterations
        # Independent sampling of the dataset
        # Need to pull the `let` trick because Julia's struggling with the capturing
        # of `total_cost` in the closure below.
        let total_cost = total_cost
            dynamic_thread(eachindex(data), 1024) do i
                # Sample this datum proportionally with its current cost
                sample_probability = samples_per_iteration * costs[i] / total_cost
                if rand() < sample_probability
                    push!(local_centroids[], data[i])
                end
            end
        end

        # Update current list of centroids
        for local_centroid_list in getall(local_centroids)
            union!(centroids, local_centroid_list)
        end

        # Recompute total cost
        compute_cost!(costs, data, centroids)
        total_cost = sum(costs)
    end

    # Choose the final candidates from this collection.
    printstyled("Finished initial run with "; color = :green, bold = true)
    print(length(centroids))
    printlnstyled(" centroids."; color = :green, bold = true)
    return refine(collect(centroids), num_centroids)
end

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
    costs = Vector{Float64}(undef, length(data))
    centroids = Set{eltype(data)}()
    push!(centroids, data[rand(1:length(data))])

    meter = ProgressMeter.Progress(num_centroids - 1, 1, "Refining Centroids ... ")
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

        # Update progress meter
        ProgressMeter.next!(meter)
    end
    ProgressMeter.finish!(meter)
    return collect(centroids)
end

#####
##### Lloyd's algorithm
#####

_minsofar() = (min_distance = typemax(Float64), min_index = 0)
function lloyds(
    data::AbstractVector{T},
    centroids::AbstractVector{T};
    max_iterations = 10,
    tol = 1E-4,
    batchsize = 1024,
) where {T}
    # As thread local storage, keep track of the nearest centroids computed so far.
    tls = ThreadLocal([(_minsofar()) for _ in 1:batchsize])

    # Accumulate points assigned to this center so far.
    # Control access with a lock.
    integrated_points = fill(zeroas(Float64, T), length(centroids))
    locks = [Base.Threads.SpinLock() for _ in 1:length(centroids)]
    points_per_center = zeros(Int, length(centroids))

    meter = ProgressMeter.Progress(max_iterations, 1, "Computing Centroids ...")
    ProgressMeter.@showprogress 1 for iter in 1:max_iterations
        dynamic_thread(batched(eachindex(data), batchsize)) do batch
            # Reset thread local storage
            min_so_far = tls[]
            min_so_far .= (_minsofar(),)

            # Compute the nearest centroids
            for center_index in 1:length(centroids), (offset, index) in enumerate(batch)
                @inbounds center = centroids[center_index]
                @unpack min_distance, min_index = min_so_far[offset]

                # Compute distance from this point to this centroid.
                # If the distance is less than the lowest so far, update the lowest.
                this_distance = distance(data[index], center)
                if this_distance < min_distance
                    update = (min_distance = this_distance, min_index = center_index)
                    min_so_far[offset] = update
                end
            end

            # Accumulate the points assigned to centroids
            # Base iteration off the batch since it could be smaller (last batch)
            for offset in eachindex(batch)
                data_index = batch[offset]
                datum = data[data_index]

                @unpack min_index = min_so_far[offset]
                Base.@lock locks[min_index] begin
                    integrated_points[min_index] += datum
                    points_per_center[min_index] += 1
                end
            end
        end

        # All cells have been integrated, compute new centroids
        new_centroids = integrated_points ./ points_per_center

        # Reset for new iteration
        zero!(integrated_points)
        zero!(points_per_center)

        # How much did we move
        movement = sum(sum.(new_centroids .- centroids))
        total = sum(sum.(centroids))

        centroids .= round.(eltype(T), new_centroids)

        # Exit condition
        relative_movement = abs(movement / total)
        ProgressMeter.next!(
            meter;
            showvalues = ((:relative_movement, relative_movement),)
        )

        (relative_movement < tol) && break
    end
    ProgressMeter.finish!(meter)

    return centroids
end
