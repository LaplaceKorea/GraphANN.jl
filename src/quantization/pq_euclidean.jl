# Specializations for Euclidean Points
# Refactored layout of PQ Centroids as follows
#
# Assume:
# - We have a 16D vector with elemets of type T
# - PQ vector sizes are 4D
# - We have 5 centroids per partition.
# - The memory layout will the be as follows
#
#         <------- N ------->
#         <-- G -->                                      Memory Layout
#            R0        R1         R2        R3
#    C0 || T T T T | T T T T || T T T T | T T T T ||       -----> Primary
#    C1 || T T T T | T T T T || T T T T | T T T T ||       |
#    C2 || T T T T | T T T T || T T T T | T T T T ||       |
#    C3 || T T T T | T T T T || T T T T | T T T T ||       ∨
#    C4 || T T T T | T T T T || T T T T | T T T T ||   Seconday
#
# The goal here is to group common numbered centroids contiguously in memory so we can do
# multiple distance computations in parallel, updating min indexes in parallel as well.
#
# Since we want to take advantage of features like AVX-512, we may need to group multiple
# centroids together into a single cache line.
#
# The sub-grouping parameter `G` will keep track of the actual sub-dimension of the PQ.
struct BinnedPQCentroids{K,G,N,T}
    centroids::NTuple{K, Vector{SIMD.Vec{N,T}}}
end

# k centroids
function binned(table::PQTable{K, <:Any, Euclidean{4, UInt8}}) where {K}
    # Make sure we can stuff this into an even number of AVX-512 vectors
    @assert iszero(mod(4 * K, 32))
    num_groups = div(4 * K, 32)
    centroids_per_group = div(32, 4)

    original_centroids = table.centroids
    num_centroids = length(first(original_centroids))

    reformatted_centroids = map(1:num_groups) do i
        range = (centroids_per_group * (i - 1) + 1):(centroids_per_group * i)

        # Slice the centroid group
        return map(1:num_centroids) do j
            pre_merged = [Tuple(original_centroids[c][j]) for c in range]
            return SIMD.Vec(_merge(pre_merged...))
        end
    end

    return BinnedPQCentroids{num_groups,4,32,UInt8}((reformatted_centroids...,))
end

Base.getindex(c::BinnedPQCentroids, i) = c.centroids[i]

# Specialize for UInt8
function unsafe_encode!(
    ptr::Ptr{UInt8},
    centroids::BinnedPQCentroids{K,4,32,UInt8},
    x::Euclidean{N,UInt8}
) where {K,N}
    return unsafe_encode!(ptr, centroids, deconstruct(SIMD.Vec{32,UInt8}, x))
end

function unsafe_encode!(
    ptr::Ptr{U},
    centroids::BinnedPQCentroids{K,4,32,UInt8},
    x::NTuple{K,SIMD.Vec{32, UInt8}}
) where {U, K}
    for i in 1:K
        v = _sub_encode_4(U, centroids[i], x[i])
        unsafe_store!(convert(Ptr{typeof(v)}, ptr), v)
        ptr += sizeof(v)
    end
end

# Convert to Int16 elements
function _sub_encode_4(
    ::Type{U},
    centroids::Vector{SIMD.Vec{32,UInt8}},
    x::SIMD.Vec{32,UInt8}
) where {U}
    return _sub_encode_4(U, centroids, convert(SIMD.Vec{32, Int16}, x))
end

function _sub_encode_4(
    ::Type{U},
    centroids::Vector{SIMD.Vec{32,UInt8}},
    x::SIMD.Vec{32,Int16}
) where {U}
    # Unlikely sad path.
    num_centroids = length(centroids)
    num_centroids > typemax(Int32) && error("Vector is too long!")

    index_type = Int32
    distance_type = Int32

    # Number of groups is `32 / 2 = 16`
    num_vnni_lanes = 16
    num_groups = 8

    minimum_distances = SIMD.Vec{num_groups, distance_type}(typemax(distance_type))
    minimum_indices = zero(SIMD.Vec{num_groups, Int32})

    for i in 1:num_centroids
        # Compute the distance between `x` and this group of centroids.
        @inbounds z = convert(SIMD.Vec{32, Int16}, centroids[i]) - x
        split_distances = _Points.vnni_accumulate(
            zero(SIMD.Vec{num_vnni_lanes, distance_type}),
            z,
            z,
        )

        # Combine every other element of the `current_distances` to get the correct
        # number of groups.
        current_distances = +(
            SIMD.shufflevector(split_distances, valtuple(Val(1), Val(2), Val(16))),
            SIMD.shufflevector(split_distances, valtuple(Val(2), Val(2), Val(16))),
        )

        # Create a SIMD mask based on which distances are lower than the minimum distances
        # so far.
        mask = current_distances < minimum_distances
        index_update = SIMD.Vec{num_groups, index_type}(i - 1)

        # Update
        minimum_indices = SIMD.vifelse(mask, index_update, minimum_indices)
        minimum_distances = SIMD.vifelse(mask, current_distances, minimum_distances)
    end
    return totuple(U, minimum_indices)
end

@generated function valtuple(
    ::Val{start},
    ::Val{step},
    ::Val{stop}
) where {start, step, stop}
tup = tuple((start-1):step:(stop-1)...)
    return :(Val($tup))
end

@generated function totuple(::Type{U}, x::SIMD.Vec{N}) where {U, N}
    exprs = [:(convert($U, x[$i])) for i in 1:N]
    return :(($(exprs...),))
end
