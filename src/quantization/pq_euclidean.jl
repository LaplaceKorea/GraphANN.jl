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
}

_centroid_dim(::Type{Euclidean{N,T}}) where {N,T} = N
_per_cacheline(::Type{<:Euclidean{<:Any,T}}) where {T} = div(64, sizeof(T))
# Since we have to expand UInt8 to Int16 - we can only store half as many per "cacheline"
_per_cacheline(::Type{<:Euclidean{<:Any,UInt8}}) = 32

# k centroids
function binned(table::PQTable{K, <:Any, T}) where {K, T <: BinCompatibleTypes}
    # Make sure we can stuff this into an even number of AVX-512 vectors
    centroid_dim = _centroid_dim(T)
    @assert iszero(mod(centroid_dim * K, _per_cacheline(T)))
    num_groups = div(centroid_dim * K, _per_cacheline(T))
    centroids_per_group = div(_per_cacheline(T), centroid_dim)

    original_centroids = table.centroids
    num_centroids = size(original_centroids, 1)

    new_centroids = map(1:num_groups) do group
        dest = Vector{SIMD.Vec{_per_cacheline(T),eltype(T)}}(undef, num_centroids)
        # Wrap the destination in a SIMDLanes to help with construction.
        lanes = SIMDLanes(SIMD.Vec{centroid_dim, eltype(T)}, dest)
        range = (centroids_per_group * (group - 1) + 1):(centroids_per_group * group)
        for (offset, i) in enumerate(range), j in 1:num_centroids
            lanes[offset, j] = original_centroids[j, i]
        end
        return dest
    end

    return BinnedPQCentroids{
        num_groups,             # Number of Macro Groups - each group expands to a cache line.
        centroids_per_group,    # Number of centroids packed in a group.
        _per_cacheline(T),      # Number of data points in a group.
        eltype(T)                   # Data encoding format.
    }(
        (new_centroids...,)
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
    U = _Points.promote_simd(E, SIMD.Vec{N,T})
    return unsafe_encode!(
        ptr,
        centroids,
        _Points.simd_wrap(U, x)
    )
end

# Stage 2-3: Promotion and storing
function unsafe_encode!(
    ptr::Ptr{P},
    centroids::BinnedPQCentroids{SLICES, GROUPS_PER_SLICE, N, T},
    x::_Points.SIMDWrap
) where {P, SLICES, GROUPS_PER_SLICE, N, T}
    for slice in 1:SLICES
        # Compute the partial result for this slice.
        partial_encoding = _sub_encode(
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

# Float32 Implementation
function _sub_encode(
    ::Val{NUM_GROUPS},
    centroids::Vector{V1},
    x::V2,
) where {NUM_GROUPS, V1 <: SIMD.Vec, V2 <: SIMD.Vec}
    # Unlikely sad path.
    index_type = Int32
    num_centroids = length(centroids)
    num_centroids > typemax(index_type) && error("Vector is too long!")

    promote_type = _Points.promote_simd(V1, V2)
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

@generated function _reduce(::Val{N}, cacheline::SIMD.Vec{16,T}) where {N,T}
    step = div(16,N)
    exprs = map(1:step) do i
        :(SIMD.shufflevector(cacheline, valtuple(Val($i), Val($step), Val(16))))
    end
    return :(reduce(+, ($(exprs...),)))
end

@generated function valtuple(
    ::Val{start},
    ::Val{step},
    ::Val{stop}
) where {start, step, stop}
tup = tuple((start-1):step:(stop-1)...)
    return :(Val($tup))
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

function _compute_distance(
    table::PQTable{NUM_GROUPS, Euclidean{N,U}, Euclidean{GROUPSIZE,T}},
    a::Euclidean{N,U},
    b::NTuple{NUM_GROUPS, <:Integer},
) where {NUM_GROUPS, N, U, GROUPSIZE, T}
    @unpack centroids = table

    # Compute some helpful constants
    centroids_per_dimension = size(centroids, 1)
    centroid_size = sizeof(Euclidean{GROUPSIZE, T})
    centroids_per_cacheline = div(64, centroid_size)
    num_cache_lines = div(NUM_GROUPS, centroids_per_cacheline)

    # Break the query into appropriate sized chunks
    #da = deconstruct(SIMD.Vec{
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