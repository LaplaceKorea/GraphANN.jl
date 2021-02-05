# # The goal here is to allow the same code to operate for full K-Means clustering OR
# # sub-K-Means clustering (i.e. product quantization).
#
# # Like the normal `Neighbor` struct, but vectorized to group together multiple individual
# # points on the same cache line.
# struct VectorNeighbor{N,I,D}
#     id::SIMD.Vec{N,I}
#     distance::SIMD.Vec{N,D}
# end
#
# function VectorNeighbor{N,I,D}() where {N,I,D}
#     return VectorNeighbor{N,I,D}(zero(SIMD.Vec{N,I}), SIMD.Vec{N,D}(typemax(D)))
# end
#
# # Scalar broadcasting.
# Base.broadcastable(x::VectorNeighbor) = (x,)
# Base.isless(a::VectorNeighbor{N}, b::VectorNeighbor{N}) where {N} = (a.distance < b.distance)
# Base.length(::VectorNeighbor{N}) where {N} = N
#
# _Base.getid(x::VectorNeighbor) = x.id
# _Base.getdistance(x::VectorNeighbor) = x.distance
#
# function update(x::VectorNeighbor{N,I,D}, d::SIMD.Vec{N,D}, i::Integer) where {N,I,D}
#     # Gets a SIMD mask with `true` indicating positions where `d` is less than
#     # `x.distance`.
#     mask = d < getdistance(x)
#     # Update `distance` and `id` to contain the smaller distance elements and their
#     # associated data ids.
#     distance = SIMD.vifelse(mask, d, x.distance)
#     id = SIMD.vifelse(mask, i, getid(x))
#     return VectorNeighbor{N,I,D}(id, distance)
# end

#####
##### Entry Points
#####

function kmeans(
    data::AbstractVector{SVector{N,T}},
    num_centroids;
    initial_oversample = 3,
    initial_iterations = 3,
    max_lloyds_iterations = 10,
) where {N,T}
    # Oversample while choosing initial centroids.
    initial = Euclidean{N,T}[]
    choose_initial_centroids!(
        data,
        initial,
        initial_oversample * num_centroids;
        num_iterations = initial_iterations,
    )

    # Refine the initial selection.
    final = Vector{SVector{N,T}}(undef, num_centroids)
    weights = parallel_count(initial, data)
    refine!(final, initial, weights)

    # Run Lloyd's algorithm.
    centroids = map(Float32, final)
    lloyds!(centroids, data; num_iterations = max_lloyds_iterations)
    return centroids
end


#####
##### Broadcasting things ...
#####

slices(x::AbstractVector) = eachindex(x)
slice(x::AbstractVector, i::Integer) = x[i]

reset(::Neighbor{T,D}) where {T,D} = Neighbor{T,D}()

#####
##### Pick Initial Centroids
#####

"""
    initial_centroids_tls(data, centroids, idtype::Type)

Allocate appropriate thread local storage for choosing initial centroids during k-means
clustering.
"""
function initial_centroids_tls(data::AbstractVector, centroids::AbstractVector, ::Type{I}) where {I}
    D = costtype(eltype(data),eltype(centroids))
    return ThreadLocal(; minimum = Neighbor{I,D}(), next = Set{I}())
end

# For the first pass, assume `data` and `centroids` are just Vectors.
function choose_initial_centroids!(
    data::AbstractArray{T},
    centroids::AbstractArray{U},
    num_centroids::Integer;
    num_iterations = 1,
    idtype::Type{I} = UInt32,
) where {T,U,I}
    # Centroids can either be supplied as a 2D array (in which case, broadcasting or
    # zipping might occur based on the type of `data`) or as a Vector, in which case
    # `data` should be a vector as well.
    @assert in(ndims(centroids), (1, 2))
    @assert in(ndims(data), (1, 2))

    # TODO: Figure out how to cleanly allocate thread local storage.
    # For example, how do we know if `costs` should be a vector or matrix.
    D = costtype(T,U)
    costs = Array{D}(undef, size(data))

    tls = initial_centroids_tls(data, centroids, idtype)
    push!(centroids, rand(data))
    adjustment = ceil(Int, num_centroids / num_iterations)

    for iteration in 1:num_iterations
        # Compute the costs for all data points.
        @time computecosts!(costs, centroids, data, tls.minimum)
        total = parallel_sum(costs)
        # Sample next points.
        parallel_sample!(costs, total, tls.next, adjustment)
        # Union updates
        next = mapreduce(x -> x.next, union, getall(tls))
        for i in next
            push!(centroids, data[i])
        end
    end
end

# Simple scalar case, just update in place.
function vupdate(minimum::Neighbor{T,D}, x, y, i::Integer; metric = Euclidean()) where {T,D}
    dist = evaluate(metric, x, y)
    new = Neighbor{T,D}(i, dist)
    return min(minimum, new)
end

# Points for Hijacking:
# - `reset` (make mutating but return the original collection)
# - `vupdate` (can also make mutating)
# - `slices` (automatic handling of mixed vectors and matrices
function findnearest!(x, Y, minimum; metric = distance)
    minimum = reset(minimum)
    for i in slices(Y)
        minimum = vupdate(minimum, x, slice(Y, i), i; metric = metric)
    end
    return minimum
end

function computecosts!(costs, centroids, data, tls; metric = distance)
    dynamic_thread(slices(data), 64) do i
        minimum = findnearest!(slice(data, i), centroids, tls[]; metric = metric)
        # TODO: generalize this cost update.
        costs[i] = getdistance(minimum)
    end
end

function refine!(final, centroids, weights)
    numfinal = length(final)
    costs = Vector{Float32}(undef, size(centroids))
    centroids_found = 1
    final[centroids_found] = rand(centroids)

    while centroids_found < numfinal
        for i in slices(centroids)
            costs[i] = minimum(
                x -> weights[i] * evaluate(Euclidean(), centroids[i], x),
                view(final, 1:centroids_found)
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
        final[centroids_found] = centroids[index]
    end
    return final
end

#####
##### Lloyd's Algorithm
#####

function lloyds!(
    centroids::AbstractArray{T},
    data;
    num_iterations = 1,
    idtype::Type{I} = UInt32
) where {T,I}
    D = costtype(T, eltype(data))
    tls = ThreadLocal(;
        minimums = Neighbor{I,D}(),
        sums = zeros(T, size(centroids)),
        points_per = zeros(Int, size(centroids)),
    )

    for iter in 1:num_iterations
        dynamic_thread(slices(data), 1024) do i
            threadlocal = tls[]
            @unpack minimums = threadlocal
            v = slice(data, i)
            neighbor = findnearest!(v, centroids, minimums)

            # Update local tracking data structures.
            # TODO: Move to own function?
            id = getid(neighbor)
            threadlocal.sums[id] += v
            threadlocal.points_per[id] += 1
        end

        sums = mapreduce(x -> x.sums, +, getall(tls))
        points_per = mapreduce(x -> x.points_per, +, getall(tls))

        for i in eachindex(sums, points_per, centroids)
            sum = sums[i]
            num_points = points_per[i]
            # TODO: Dummy check for now.
            # Need to determine the correct thing to do if a centroid has no assigned
            # points.
            @assert !iszero(num_points)
            centroids[i] = sum / num_points
        end
    end
end

#####
##### Parallel Algorithms
#####

"""
    parallel_sum(x::AbstractArray) -> Vector

Return the sum of each column of `x`.
"""
function parallel_sum(x::AbstractVector{T}) where {T}
    tls = ThreadLocal(Ref(zero(maybe_widen(T))))
    dynamic_thread(eachindex(x), 64) do i
        localtotal = tls[]
        localtotal[] += x[i]
    end
    return sum(getindex, getall(tls))
end

"""
    parallel_count(centroids, data) -> Array{Int}

Return an array of `size(centroids)` with the counts of how many points in `data` are
closest to each centroid.
"""
function parallel_count(centroids, data)
    D = costtype(eltype(centroids), eltype(data))
    tls = ThreadLocal(;
        minimum = Neighbor{UInt32,D}(),
        counts = zeros(Int, size(centroids)),
    )

    dynamic_thread(slices(data), 64) do i
        threadlocal = tls[]
        minimum = findnearest!(slice(data, i), centroids, threadlocal.minimum)
        threadlocal.counts[getid(minimum)] += 1
    end
    return mapreduce(x -> x.counts, +, getall(tls))
end

function parallel_sample!(
    costs::AbstractVector,
    totalcost::Number,
    tls::ThreadLocal{<:Set},
    adjustment::Number
)
    foreach(empty!, getall(tls))
    dynamic_thread(slices(costs), 64) do i
        next = tls[]
        val = slice(costs, i)
        sample_probability = adjustment * val / totalcost
        if rand() < sample_probability
            push!(next, i)
        end
    end
end

