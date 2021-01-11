force_store!(ptr::Ptr{T}, x::T) where {T} = unsafe_store!(ptr, x)
force_store!(ptr::Ptr{T}, x::U) where {T,U} = force_store!(Ptr{U}(ptr), x)

#####
##### Specialize for Euclidean
#####

# When we can stuff multiple centroids on a single cacheline, use `Packed` to represent
# this, allowing us to compute multiple matchings at a time.
struct BinnedPQCentroids{K, P <: Packed}
    centroids::NTuple{K, Vector{P}}
end

encoded_length(::BinnedPQCentroids{K, P}) where {K, P} = K * length(P)

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

    # Sanity Checks
    # Make sure we can stuff this into an even number of AVX-512 vectors.
    @assert iszero(mod(original_length, centroids_per_cacheline))
    @assert num_partitions == size(centroids, 2)

    packed_centroids = ntuple(Val(num_cachelines)) do cacheline
        start = centroids_per_cacheline * (cacheline - 1) + 1
        stop = centroids_per_cacheline * cacheline
        range = start:stop

        return map(1:num_centroids) do centroid
            vals = [centroids[centroid, partition_number] for partition_number in range]
            Packed(vals...)
        end
    end

    return BinnedPQCentroids(packed_centroids)
end
Base.getindex(c::BinnedPQCentroids, i) = c.centroids[i]

# Type conversion pipeline.
#
# 1. Split argument `x` into a tuple of SIMD.Vec sized pieces.
# 2. Call `partial_encode` on each cacheline(ish) sized chunk, promoting the slices of `x`
#    to the correct type.
# 3. Sequentially store results to the destination pointer.

# Stage 1: Deconstruction
function unsafe_encode!(
    ptr::Ptr,
    centroids::BinnedPQCentroids{<:Any,P},
    x::E
) where {P,E <: Euclidean}
    U = _Points.distance_type(E, _Points.distance_type(P))
    return unsafe_encode!(
        ptr,
        centroids,
        _Points.EagerWrap{U}(x)
    )
end

# Stage 2-3: Promotion and storing
function unsafe_encode!(
    ptr::Ptr{U},
    centroids::BinnedPQCentroids{SLICES, P},
    x::_Points.EagerWrap,
) where {U, SLICES, P}
    for slice in 1:SLICES
        # Compute the partial result for this slice.
        partial_encoding = partial_encode(
            centroids[slice],
            P(x[slice]),
        )

        # Store to pointer
        converted_encoding = convert.(U, partial_encoding)
        force_store!(ptr, converted_encoding)
        ptr += sizeof(converted_encoding)
    end
end

# TODO: Reformulate using `CurrentMinimum`..
function partial_encode(
    centroids::Vector{P},
    x::P,
) where {K, E, V, P <: Packed{K,E,V}}
    # Unlikely sad path.
    index_type = Int32
    num_centroids = length(centroids)
    num_centroids > typemax(index_type) && error("Vector is too long!")

    promote_type = _Points.distance_type(V)
    distance_type = eltype(_Points.accum_type(promote_type))

    # Keep track of one per group
    minimum = CurrentMinimum{K, distance_type}()
    for i in 1:num_centroids
        current_distances = distance(x, centroids[i])
        minimum = update(minimum, current_distances, i - 1)
    end
    return Tuple(minimum.index)
end

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
        @inbounds _b = convert(promote_type, centroids[maybe_widen(b[i]) + one(maybe_widen(I)), i])
        z = _a - _b
        accumulator = _Points.square_accum(z, accumulator)
    end
    return sum(accumulator)
end

