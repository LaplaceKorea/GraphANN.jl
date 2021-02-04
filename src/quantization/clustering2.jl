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
    return ThreadLocal(; neighbors = Neighbor{I,D}(), next = Set{I}())
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
        computecosts!(costs, centroids, data, tls)
        total = parallelsum(costs)
        # Sample next points.
        sample!(costs, total, tls, adjustment)
        # Union updates
        next = mapreduce(x -> x.next, union, getall(tls))
        for i in next
            push!(centroids, data[i])
        end
        println()
    end
end

function parallelsum(costs::AbstractVector{T}) where {T}
    tls = ThreadLocal(Ref(zero(maybe_widen(T))))
    dynamic_thread(eachindex(costs), 64) do i
        localtotal = tls[]
        localtotal[] += costs[i]
    end
    return sum(getindex, getall(tls))
end

# Simple scalar case, just update in place.
function vupdate(neighbor::Neighbor{T,D}, x, y, i::Integer) where {T,D}
    newdistance = distance(x, y)
    newneighbor = Neighbor{T,D}(i, newdistance)
    return min(neighbor, newneighbor)
end

function computecosts!(costs, centroids, data, tls)
    dynamic_thread(slices(data), 64) do i
        neighbors = findneighbors!(slice(data, i), centroids, tls[].neighbors)
        # TODO: generalize this cost update.
        costs[i] = getdistance(neighbors)
    end
end

# Points for Hijacking:
# - `reset` (make mutating but return the original collection)
# - `vupdate` (can also make mutating)
# - `slices` (automatic handling of mixed vectors and matrices
function findneighbors!(x, Y, neighbors)
    neighbors = reset(neighbors)
    for i in slices(Y)
        neighbors = vupdate(neighbors, x, slice(Y, i), i)
    end
    return neighbors
end

function sample!(costs::AbstractVector, totalcost::Number, tls, adjustment::Number)
    foreach(x -> empty!(x.next), getall(tls))
    dynamic_thread(slices(costs), 64) do i
        next = tls[].next
        val = slice(costs, i)
        sample_probability = adjustment * val / totalcost
        if rand() < sample_probability
            push!(next, i)
        end
    end
end
