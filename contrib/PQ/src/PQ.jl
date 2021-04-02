module PQ

import GraphANN: GraphANN, MaybeThreadLocal

# stdlib
using Statistics

# deps
using SIMD: SIMD
import StaticArrays: SVector
using ProgressMeter: ProgressMeter
import UnPack: @unpack

####
##### Product Quantize a dataset
#####

function exact_div(A, B)
    div, rem = divrem(A, B)
    if iszero(rem)
        return div
    else
        return error("Data length $A is not divisible by the number of partitions $B")
    end
end

# Internally, we try to work with full-width cacheline vectors as much as possible,
# using SIMD parallelism.
#
# This works with the assumption that the size of the centroids is fairly small, think
# SVector{4,Float32} or SVector{8,Float32}
# As such, we can perform 4 or 2 distance computations in parallel.
const BroadcastVector = SVector{16,Float32}

# For now - hardcode centroid eltypes as `Float32`.
# This always seems to perform better than keeping the centroids as integers anyways
# so we don't lose much flexibility with this.
struct DistanceTable{N}
    # Stored in a column major form.
    # - `centroids[1,1]` returns the first centroid for the first partition.
    # - `centroids[2,1]` returns the second centroid for the first partition.
    centroids::Matrix{SVector{N,Float32}}
    # Cached distance table
    distances::Matrix{Float32}

    # Use an inner constructor to ensure that the distance `table` is always the same
    # size as `centroids`.
    function DistanceTable(centroids::Matrix{SVector{N,Float32}}) where {N}
        distances = Matrix{Float32}(undef, size(centroids))
        return new{N}(centroids, distances)
    end
end

GraphANN.costtype(::MaybeThreadLocal{DistanceTable}, ::AbstractVector{<:NTuple}) = Float32

# For the thread copy - keep the centroids the same to slightly reduce memory.
GraphANN._Base.threadcopy(x::DistanceTable) = DistanceTable(x.centroids)

# Broadcast to a cacheline
broadcast(x::SVector{N,T}) where {N,T} = broadcast(GraphANN.toeltype(Float32, x))
broadcast(x::SVector{16,Float32}) = x
function broadcast(x::SVector{N,Float32}) where {N}
    # How many can we fit onto a cacheline.
    replications = exact_div(16, N)
    return vcat(ntuple(_ -> x, Val(replications))...)
end

sum_every(x::SVector, valn::Val) = SVector(Tuple(sum_every(SIMD.Vec(Tuple(x)), valn)))
@generated function sum_every(x::SIMD.Vec{S,T}, ::Val{N}) where {S,T,N}
    # Otherwise, build up a reduction tree.
    exprs = map(1:N) do i
        tup = tuple(i:N:S...) .- 1
        :(SIMD.shufflevector(x, Val($tup)))
    end
    return quote
        Base.@_inline_meta
        reduce(+, ($(exprs...),))
    end
end

force_store!(ptr::Ptr{T}, x::T) where {T} = unsafe_store!(ptr, x)
force_store!(ptr::Ptr{T}, x::U) where {T,U} = force_store!(Ptr{U}(ptr), x)

force_load(::Type{T}, ptr::Ptr{T}) where {T} = unsafe_load(ptr)
force_load(::Type{T}, ptr::Ptr) where {T} = unsafe_load(Ptr{T}(ptr))

# Below function is unsafe because it assumes the sizes of `src` is an exact multiple
# of the size of `point` and that the destination is valid.
#
# These invariants must be maintained within the top level `DistanceTable`.
@inline function store_distances!( # Not a safe function
    dest::AbstractVector{Float32},
    src::AbstractVector{SVector{N,Float32}},
    point::SVector{N},
) where {N}
    point_broadcast = broadcast(point)
    dst_ptr = pointer(dest)
    src_ptr = pointer(src)

    src_stop = src_ptr + sizeof(src)
    while src_ptr < src_stop
        # Compute square Euclidean distances using SIMD parallelism.
        src_point = force_load(typeof(point_broadcast), src_ptr)
        distances = sum_every((src_point - point_broadcast) .^ 2, Val(N))
        force_store!(dst_ptr, distances)

        src_ptr += sizeof(point_broadcast)
        dst_ptr += sizeof(distances)
    end
    return nothing
end

function _getindex(x::SVector, ::Val{N}, i::Integer) where {N}
    return SVector(ntuple(j -> @inbounds(getindex(x, N * (i - 1) + j)), Val(N)))
end

function precompute!(table::DistanceTable{N}, query::SVector) where {N}
    @unpack centroids, distances = table
    num_partitions = size(centroids, 2)

    @assert N * size(table.centroids, 2) == length(query)
    for i in Base.OneTo(num_partitions)
        @inbounds store_distances!(
            view(distances, :, i), view(centroids, :, i), _getindex(query, Val(N), i)
        )
    end
end

function GraphANN.prehook(table::MaybeThreadLocal{DistanceTable}, query::SVector)
    precompute!(GraphANN.getlocal(table), query)
    return nothing
end

const MaybePtr{T} = Union{T,Ptr{<:T}}
maybeload(x) = x
maybeload(ptr::Ptr) = unsafe_load(ptr)

@inline function GraphANN.evaluate(
    table::MaybeThreadLocal{DistanceTable}, ::SVector, x::MaybePtr{NTuple}
)
    return lookup(GraphANN.getlocal(table), maybeload(x))
end

# N.B.: The indices in `inds` are index-0 to take full advantage of the range offered by UInt8's.
@inline function lookup(table::DistanceTable, inds::NTuple{K}) where {K}
    @unpack distances = table
    s0 = zero(Float32)
    s1 = zero(Float32)
    s2 = zero(Float32)
    s3 = zero(Float32)
    i = 1

    # Unroll 4 times as long as possible.
    @inbounds while i + 3 <= K
        j0 = Int(inds[i + 0]) + 1
        j1 = Int(inds[i + 1]) + 1
        j2 = Int(inds[i + 2]) + 1
        j3 = Int(inds[i + 3]) + 1

        s0 += distances[j0, i + 0]
        s1 += distances[j1, i + 1]
        s2 += distances[j2, i + 2]
        s3 += distances[j3, i + 3]

        i += 4
    end

    # Catch the remainders
    @inbounds while i <= K
        s0 += distances[Int(inds[i]) + 1, i]
        i += 1
    end

    return s0 + s1 + s2 + s3
end

function encode(
    _table::DistanceTable{N},
    data::AbstractVector{<:SVector{K}},
    itype::Type{I};
    allocator = GraphANN.stdallocator,
    executor = GraphANN.dynamic_thread,
) where {N,K,I}
    # Copy one for each thread
    tables = GraphANN.ThreadLocal(_table)

    # Number of indices in the tuple.
    tuple_dim = exact_div(K, N)
    dest = allocator(NTuple{tuple_dim,I}, size(data)...)

    batchsize = 2048
    meter = ProgressMeter.Progress(length(data), 1, "Converting Dataset")

    executor(GraphANN.batched(eachindex(data), batchsize)) do range
        for i in range
            table = GraphANN.getlocal(tables)
            # Compute distance from this data point to all centroids.
            precompute!(table, data[i])
            @unpack distances = table
            ptr = pointer(dest, i)
            for j in 1:tuple_dim
                _, min_ind = findmin(view(distances, :, j))
                force_store!(ptr, convert(I, min_ind - one(min_ind)))
                ptr += sizeof(I)
            end
        end
        ProgressMeter.next!(meter; step = batchsize)
    end
    return dest
end

function fullrecall(ids, groundtruth, num_neighbors)
    function intersect_length((x, y),)
        vy = view(y, 1:num_neighbors)
        return length(intersect(x, vy))
    end
    overlap = sum(intersect_length, zip(eachcol(ids), eachcol(groundtruth)))
    return overlap / (num_neighbors * size(ids, 2))
end

end # module
