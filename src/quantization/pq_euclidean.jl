# Specializations for Euclidean Points
# Refactored layout of PQ Centroids as follows
#
# Assume:
# - We have a 16D vector with elemets of type T
# - PQ vector sizes are 4D
# - We have 5 centroids per partition.
# - The memory layout will the be as follows
#
#             <------- N ------->
#                 +----G----+
#                 |         |
#                 ∨         ∨           Memory order
#                R0        R1
#        C0 || T T T T | T T T T ||       -----> Primary
#        C1 || T T T T | T T T T ||       |
#   +--> C2 || T T T T | T T T T ||       |
#   |    C3 || T T T T | T T T T ||       ∨
#   |    C4 || T T T T | T T T T ||   Seconday
#   |
#   K
#   |            R2        R3
#   |    C0 || T T T T | T T T T ||
#   |    C1 || T T T T | T T T T ||
#   +--> C2 || T T T T | T T T T ||
#        C3 || T T T T | T T T T ||
#        C4 || T T T T | T T T T ||
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

encoded_type(::Type{U}, ::BinnedPQCentroids{K, G}) where {U, K, G} = NTuple{K*G, U}

const BinCompatibleTypes = Union{
    Euclidean{4,UInt8},
    Euclidean{8,UInt8},
}

_centroid_dim(::Type{Euclidean{N,T}}) where {N,T} = N

# k centroids
function binned(table::PQTable{K, <:Any, T}) where {K, T <: BinCompatibleTypes}
    # Make sure we can stuff this into an even number of AVX-512 vectors
    centroid_dim = _centroid_dim(T)
    @assert iszero(mod(centroid_dim * K, 32))
    num_groups = div(centroid_dim * K, 32)
    centroids_per_group = div(32, centroid_dim)

    original_centroids = table.centroids
    num_centroids = size(original_centroids, 1)

    new_centroids = map(1:num_groups) do group
        dest = Vector{SIMD.Vec{32,UInt8}}(undef, num_centroids)
        # Wrap the destination in a SIMDLanes to help with construction.
        lanes = SIMDLanes(SIMD.Vec{centroid_dim, UInt8}, dest)
        range = (centroids_per_group * (group - 1) + 1):(centroids_per_group * group)
        for (offset, i) in enumerate(range), j in 1:num_centroids
            lanes[offset, j] = original_centroids[j, i]
        end
        return dest
    end

    return BinnedPQCentroids{
        num_groups,             # Number of Macro Groups - each group expands to a cache line.
        centroids_per_group,    # Number of centroids packed in a group.
        32,                     # Number of data points in a group.
        UInt8                   # Data encoding format.
    }(
        (new_centroids...,)
    )
end

Base.getindex(c::BinnedPQCentroids, i) = c.centroids[i]

_merge(x, y, z...) = (x..., _merge(y, z...)...)
_merge(x, y) = (x..., y...)
_merge(x) = x

# Specialize for UInt8
function unsafe_encode!(
    ptr::Ptr{UInt8},
    centroids::BinnedPQCentroids{<:Any,<:Any,32,UInt8},
    x::Euclidean{<:Any,UInt8}
)
    return unsafe_encode!(ptr, centroids, deconstruct(SIMD.Vec{32,UInt8}, x))
end

function unsafe_encode!(
    ptr::Ptr{U},
    centroids::BinnedPQCentroids{K,N,32,UInt8},
    x::NTuple{K,SIMD.Vec{32, UInt8}}
) where {U, K, N}
    for i in 1:K
        v = _sub_encode(Val(N), U, centroids[i], convert(SIMD.Vec{32,Int16}, x[i]))
        unsafe_store!(convert(Ptr{typeof(v)}, ptr), v)
        ptr += sizeof(v)
    end
end

# Specialization for 8 groups per cacheline
function _sub_encode(
    ::Val{8},
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

# Specialization for 4 groups per cacheline
function _sub_encode(
    ::Val{4},
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
    num_groups = 4

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

        # Distance Reduction.
        current_distances = +(
            SIMD.shufflevector(split_distances, valtuple(Val(1), Val(4), Val(16))),
            SIMD.shufflevector(split_distances, valtuple(Val(2), Val(4), Val(16))),
            SIMD.shufflevector(split_distances, valtuple(Val(3), Val(4), Val(16))),
            SIMD.shufflevector(split_distances, valtuple(Val(4), Val(4), Val(16))),
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

#####
##### Distance Computations
#####

# Specialize
function (table::PQTable{K,Euclidean{N,UInt8},T})(
    a::Euclidean{N,UInt8},
    b::NTuple{K, <:Integer}
) where {K,N, T <: BinCompatibleTypes}
    return _compute_distance(table, a, b)
end

# Specialze for 8 centroids per cache line (4 elements per centroid)
function _compute_distance(
    table::PQTable{K,Euclidean{N,UInt8},Euclidean{4,UInt8}},
    a::Euclidean{N,UInt8},
    b::NTuple{K, <:Integer}
) where {K,N}
    @unpack centroids = table

    centroids_per_dimension = size(centroids, 1)
    centroids_per_cacheline = 8
    centroid_size = sizeof(Euclidean{4,UInt8})
    num_cache_lines = 2 * (N * sizeof(UInt8)) >> 6

    # Break argument `a` into half-cacheline sized chunks.
    # This is because we will soon promote the UInt8 elements to Int16 to perform correct
    # arithmetic.
    # We want these Int16 elements to consume a whole cacheline.
    da = deconstruct(SIMD.Vec{32,UInt8}, a)
    accumulator = zero(SIMD.Vec{16,Int32})

    # Use a pointer to Int32's since we're fetching 4 UInt8's at a time, which is
    # the same size as an Int32.
    base_ptr = convert(Ptr{Int32}, pointer(centroids))
    for i in 0:(num_cache_lines - 1)
        base_ptr_adjusted = +(
            base_ptr,
            i * centroid_size * centroids_per_dimension * centroids_per_cacheline
        )

        # Construct the `b` vector using a `gather` instruction.
        pointers = ntuple(Val(centroids_per_cacheline)) do j
            full_index = i * centroids_per_cacheline + j
            return base_ptr_adjusted + centroid_size *
                (((j - 1) * centroids_per_dimension) + b[full_index])
        end

        # Here is where we convert the implicitly gathered groups of four UInt8's back
        # from their packed Int32 form.
        vb = reinterpret(SIMD.Vec{32, UInt8}, SIMD.vgather(SIMD.Vec(pointers)))
        converted_da = convert(SIMD.Vec{32, Int16}, da[i + 1])
        converted_vb = convert(SIMD.Vec{32, Int16}, vb)
        z = converted_da - converted_vb

        accumulator = _Points.vnni_accumulate(accumulator, z, z)
    end
    return sum(accumulator)
end

function _compute_distance(
    table::PQTable{K,Euclidean{N,UInt8},Euclidean{8,UInt8}},
    a::Euclidean{N,UInt8},
    b::NTuple{K, <:Integer}
) where {K,N}
    @unpack centroids = table

    centroids_per_dimension = size(centroids, 1)
    centroids_per_cacheline = 4
    centroid_size = sizeof(Euclidean{8,UInt8})
    num_cache_lines = 2 * (N * sizeof(UInt8)) >> 6

    # Break argument `a` into half-cacheline sized chunks.
    # This is because we will soon promote the UInt8 elements to Int16 to perform correct
    # arithmetic.
    # We want these Int16 elements to consume a whole cacheline.
    da = deconstruct(SIMD.Vec{32,UInt8}, a)
    accumulator = zero(SIMD.Vec{16,Int32})

    # Use a pointer to Int32's since we're fetching 4 UInt8's at a time, which is
    # the same size as an Int32.
    base_ptr = convert(Ptr{Int64}, pointer(centroids))
    for i in 0:(num_cache_lines - 1)
        base_ptr_adjusted = +(
            base_ptr,
            i * centroid_size * centroids_per_dimension * centroids_per_cacheline
        )

        # Construct the `b` vector using a `gather` instruction.
        pointers = ntuple(Val(centroids_per_cacheline)) do j
            full_index = i * centroids_per_cacheline + j
            return base_ptr_adjusted + centroid_size *
                (((j - 1) * centroids_per_dimension) + b[full_index])
        end

        # Here is where we convert the implicitly gathered groups of four UInt8's back
        # from their packed Int32 form.
        vb = reinterpret(SIMD.Vec{32, UInt8}, SIMD.vgather(SIMD.Vec(pointers)))
        converted_da = convert(SIMD.Vec{32, Int16}, da[i + 1])
        converted_vb = convert(SIMD.Vec{32, Int16}, vb)
        z = converted_da - converted_vb

        accumulator = _Points.vnni_accumulate(accumulator, z, z)
    end
    return sum(accumulator)
end
