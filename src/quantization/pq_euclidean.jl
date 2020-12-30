force_store!(ptr::Ptr{T}, x::T) where {T} = unsafe_store!(ptr, x)
force_store!(ptr::Ptr{T}, x::U) where {T,U} = force_store!(Ptr{U}(ptr), x)

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

encoded_length(::BinnedPQCentroids{K, G}) where {K, G} = K*G

const BinCompatibleTypes = Union{
    Euclidean{4,UInt8},
    Euclidean{8,UInt8},
    Euclidean{4,Float32},
    Euclidean{8,Float32},
    Euclidean{16,Float32},
}

_centroid_length(::Type{Euclidean{N,T}}) where {N,T} = N
_per_cacheline(::Type{<:Euclidean{<:Any,T}}) where {T} = div(64, sizeof(T))
# Since we have to expand UInt8 to Int16 - we can only store half as many per "cacheline"
_per_cacheline(::Type{<:Euclidean{<:Any,UInt8}}) = 32

function binned(table::PQTable{K, T}) where {K, T <: BinCompatibleTypes}
    @unpack centroids = table

    # The goal here is to repack neighboring centroids into a single cache line.
    # This allows several centroids to be encoded at a time during the encoding phase.
    num_centroids = size(centroids, 1)
    num_partitions = K
    centroid_length = _centroid_length(T)
    original_length = centroid_length * num_partitions
    points_per_cacheline = _per_cacheline(T)
    centroids_per_cacheline = div(points_per_cacheline, centroid_length)
    num_cachelines = div(num_partitions, centroids_per_cacheline)
    data_type = eltype(T)

    # Sanity Checks
    # Make sure we can stuff this into an even number of AVX-512 vectors.
    @assert iszero(mod(original_length, centroids_per_cacheline))
    @assert num_partitions == size(centroids, 2)

    # Here, we define two different types.
    # - `store_type` is the packed SIMD representation, storing multiple centroids
    # per cache line.
    # - `move_type` is a SIMD vector for a single centroid vector. This type should be
    # strictly smaller than `store_type` above and is used to pack centroid data into
    # their corresponding locations in the packed representation.
    store_type = SIMD.Vec{points_per_cacheline, data_type}
    move_type = eltype(centroids)

    packed_centroids = map(1:num_cachelines) do cacheline
        dest = Vector{store_type}(undef, num_centroids)

        # To help with the packing, temporarily
        lanes = reinterpret(move_type, dest) |> (x -> reshape(x, centroids_per_cacheline, :))

        start = centroids_per_cacheline * (cacheline - 1) + 1
        stop = centroids_per_cacheline * cacheline
        range = start:stop

        for (lane_offset, partition_number) in enumerate(range), centroid in 1:num_centroids
            lanes[lane_offset, centroid] = centroids[centroid, partition_number]
        end
        return dest
    end

    return BinnedPQCentroids{
        num_cachelines,           # Number of Macro Groups - each group expands to a cache line.
        centroids_per_cacheline,  # Number of centroids packed in a group.
        points_per_cacheline,     # Number of data points in a group.
        data_type,                # Data encoding format.
    }(
        (packed_centroids...,)
    )
end
Base.getindex(c::BinnedPQCentroids, i) = c.centroids[i]

# Type conversion pipeline.
#
# 1. Split argument `x` into a tuple of SIMD.Vec sized pieces.
# 2. Call `_sub_encode` on each cacheline(ish) sized chunk, promoting the slices of `x`
#    to the correct type.
# 3. Sequentially store results to the destination pointer.

# Stage 1: Deconstruction
function unsafe_encode!(
    ptr::Ptr,
    centroids::BinnedPQCentroids{<:Any,<:Any,N,T},
    x::E
) where {N,T,E <: Euclidean}
    U = _Points.distance_type(E, SIMD.Vec{N,T})
    return unsafe_encode!(
        ptr,
        centroids,
        _Points.EagerWrap{U}(x)
    )
end

# Stage 2-3: Promotion and storing
function unsafe_encode!(
    ptr::Ptr{P},
    centroids::BinnedPQCentroids{SLICES, GROUPS_PER_SLICE, N, T},
    x::_Points.EagerWrap,
) where {P, SLICES, GROUPS_PER_SLICE, N, T}
    for slice in 1:SLICES
        # Compute the partial result for this slice.
        partial_encoding = partial_encode(
            Val(GROUPS_PER_SLICE),
            centroids[slice],
            x[slice],
        )

        # Store to pointer
        converted_encoding = convert.(P, partial_encoding)
        force_store!(ptr, converted_encoding)
        ptr += sizeof(converted_encoding)
    end
end

function partial_encode(
    ::Val{NUM_GROUPS},
    centroids::Vector{V1},
    x::V2,
) where {NUM_GROUPS, V1 <: SIMD.Vec, V2 <: SIMD.Vec}
    # Unlikely sad path.
    index_type = Int32
    num_centroids = length(centroids)
    num_centroids > typemax(index_type) && error("Vector is too long!")

    promote_type = _Points.distance_type(V1, V2)
    distance_type = eltype(_Points.accum_type(promote_type))

    # Keep track of one per group
    minimum_distances = SIMD.Vec{NUM_GROUPS, distance_type}(typemax(distance_type))
    minimum_indices = zero(SIMD.Vec{NUM_GROUPS, index_type})

    _x = convert(promote_type, x)
    for i in 1:num_centroids
        c = @inbounds convert(promote_type, centroids[i])
        z = _Points.square(c - _x)
        current_distances = _reduce(Val(NUM_GROUPS), z)

        # Create a SIMD mask based on which distances are lower than the minimum distances
        # so far.
        mask = current_distances < minimum_distances
        index_update = SIMD.Vec{NUM_GROUPS, index_type}(i - 1)

        # Update
        minimum_indices = SIMD.vifelse(mask, index_update, minimum_indices)
        minimum_distances = SIMD.vifelse(mask, current_distances, minimum_distances)
    end
    return Tuple(minimum_indices)
end

# TODO: Move this into "euclidean.jl"?
@generated function _reduce(::Val{N}, cacheline::SIMD.Vec{S,T}) where {N,S,T}
    step = div(S,N)
    exprs = map(1:step) do i
        tup = valtuple(i, step, S)
        :(SIMD.shufflevector(cacheline, Val($tup)))
    end
    return :(reduce(+, ($(exprs...),)))
end

valtuple(start, step, stop) = tuple((start - 1):step:(stop - 1)...)

#####
##### Distance Computations
#####

# Specialize
function (table::PQTable{K,E})(
    a::Euclidean{N,T},
    b::NTuple{K, <:Integer}
) where {K, E <: Euclidean, N, T}
    return compute_distance(table, a, b)
end

function pq_distance_type(::Type{Euclidean{N,Float32}}, ::Type{<:Euclidean}) where {N}
    SIMD.Vec{N,Float32}
end

function (table::PQTable{K, Euclidean{N1,T1}})(
    a::Euclidean{N2,T2},
    b::NTuple{K, I},
) where {K, N1, N2, T1, T2, I <: Integer}
    Base.@_inline_meta

    @unpack centroids = table
    promote_type = pq_distance_type(Euclidean{N1,T1}, Euclidean{N2,T2})
    da = _Points.LazyWrap{promote_type}(a)
    accumulator = zero(_Points.accum_type(promote_type))

    for i in 1:K
        @inbounds _a = da[i]
        # TODO: Properly handle conversion
        @inbounds _b = convert(promote_type, centroids[b[i] + one(I), i])
        z = _a - _b
        accumulator = _Points.square_accum(z, accumulator)
    end
    return sum(accumulator)
end

