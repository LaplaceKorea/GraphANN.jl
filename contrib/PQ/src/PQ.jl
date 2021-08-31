module PQ

import GraphANN: GraphANN, MaybeThreadLocal

# stdlib
using Statistics

# deps
using SIMD: SIMD
import StaticArrays: StaticVector, SVector, MVector
using ProgressMeter: ProgressMeter
import UnPack: @unpack

include("compress.jl")

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
struct DistanceTable{N,M}
    # Stored in a column major form.
    # - `centroids[1,1]` returns the first centroid for the first partition.
    # - `centroids[2,1]` returns the second centroid for the first partition.
    centroids::Matrix{SVector{N,Float32}}
    # Cached distance table
    distances::Matrix{Float32}
    metric::M

    # Use an inner constructor to ensure that the distance `table` is always the same
    # size as `centroids`.
    function DistanceTable(
        centroids::Matrix{SVector{N,Float32}}, metric::M = GraphANN.Euclidean();
    ) where {N,M}
        distances = Matrix{Float32}(undef, size(centroids))
        return new{N,M}(centroids, distances, metric)
    end
end

# Help out type inference - we'll always end up in Float32 land anyways.
GraphANN.costtype(::MaybeThreadLocal{DistanceTable}, args...) = Float32
GraphANN.costtype(::MaybeThreadLocal{DistanceTable}, ::AbstractVector) = Float32
GraphANN.costtype(::MaybeThreadLocal{DistanceTable}, ::Type{<:Any}) = Float32
GraphANN.costtype(::MaybeThreadLocal{DistanceTable}, ::Type{<:Any}, ::Type{<:Any}) = Float32

# For the thread copy - keep the centroids the same to slightly reduce memory.
GraphANN._Base.threadcopy(x::DistanceTable) = DistanceTable(x.centroids, x.metric)

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
    metric,
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
    @unpack centroids, distances, metric = table
    num_partitions = size(centroids, 2)

    @assert N * size(table.centroids, 2) == length(query)
    for i in Base.OneTo(num_partitions)
        @inbounds store_distances!(
            metric,
            view(distances, :, i),
            view(centroids, :, i),
            _getindex(query, Val(N), i),
        )
    end
end

maybeload(x) = x
maybeload(ptr::Ptr) = unsafe_load(ptr)

function GraphANN.prehook(table::DistanceTable, query::GraphANN.MaybePtr{SVector})
    precompute!(GraphANN.getlocal(table), maybeload(query))
    return nothing
end

@inline function GraphANN.evaluate(
    table::DistanceTable, ::GraphANN.MaybePtr{SVector}, x::GraphANN.MaybePtr{NTuple}
)
    return lookup(table, maybeload(x))
end

# The generated lookup basically unrolls the entire distance lookup.
# It seems to be slightly faster than having a loop.
#
# However, we may need to fall back to a loop for large "K".
# If performance degrades for datasets with high dimensionality, we'll revisit this.
@generated function lookup(table::DistanceTable, inds::NTuple{K}) where {K}
    return _lookup_impl(K)
end

_gensym(i; prefix = "s") = Symbol("$(prefix)_$(i)")
function _lookup_impl(K)
    loads = [:($(_gensym(i)) = @inbounds(Int(inds[$i]))) for i in 1:K]
    incr = [:($(_gensym(i; prefix = "j")) = $(_gensym(i)) + 1) for i in 1:K]
    exprs = map(1:K) do i
        :($(_gensym(i; prefix = "k")) = @inbounds(distances[$(_gensym(i; prefix = "j")), $i]))
    end
    syms = [:($(_gensym(i; prefix = "k"))) for i in 1:K]

    return quote
        Base.@_inline_meta
        @unpack distances = table
        $(loads...)
        $(incr...)
        $(exprs...)
        return sum(($(syms...),))
    end
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
            # Compute distance from this data point to all centroids.
            # Then iterate over each partition, finding the best match within that partition
            # and storing it to the final result.
            table = GraphANN.getlocal(tables)
            precompute!(table, data[i])
            @unpack distances = table
            ptr = pointer(dest, i)

            # Iterating over partitions
            for j in Base.OneTo(tuple_dim)
                # Iterating down distances to centroids within the partition.
                _, min_ind = findmin(view(distances, :, j))
                force_store!(ptr, convert(I, min_ind - one(min_ind)))
                ptr += sizeof(I)
            end
        end
        ProgressMeter.next!(meter; step = batchsize)
    end
    return dest
end

#####
##### Post Processing
#####

struct Reranker{A<:AbstractVector,M}
    dataset::A
    metric::M
end

function Reranker(dataset::AbstractVector; metric = GraphANN.Euclidean())
    return Reranker(dataset, metric)
end

function (reranker::Reranker)(
    runner::GraphANN.Algorithms.DiskANNRunner, num_neighbors, query
)
    @unpack dataset, metric = reranker
    @unpack buffer = runner
    @unpack entries = buffer

    # Populate the full distance for each candidate.
    for i in eachindex(entries)
        id = GraphANN.getid(entries[i])
        distance = GraphANN.evaluate(metric, query, pointer(dataset, id))
        @inbounds GraphANN.Algorithms.unsafe_replace!(buffer, i, id, distance)
    end

    # Sort based on distance.
    return partialsort!(entries, Base.OneTo(num_neighbors))
end

#####
##### IO
#####

function load_diskann_centroids(path::AbstractString, args...; kw...)
    return open(path; read = true) do io
        load_diskann_centroids(io, args...; kw...)
    end
end

# DiskANN has an offset that they apply to the data points for some reason.
# Perhaps it helps when product quantizing InnerProduct based
function load_diskann_centroids(io::IO, fulldim, centroiddim; offsetpath = nothing)
    pre_centroids = GraphANN.load_bin(GraphANN.DiskANN(), SVector{fulldim,Float32}, io)
    if offsetpath !== nothing
        offset = SVector{fulldim,Float32}(
            GraphANN.load_bin(GraphANN.DiskANN(), Float32, offsetpath)
        )
        pre_centroids .+= Ref(offset)
    end
    ncentroids = length(pre_centroids)
    centroids = collect(
        permutedims(
            reinterpret(reshape, SVector{centroiddim,Float32}, pre_centroids), (2, 1)
        ),
    )

    # Maybe get an offset as well.
    return centroids
end

end # module
