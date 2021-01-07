#####
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

# Type Parameters
# N - size(self.centroids, 2) - The number of centroids in this clustering.
# T - The element type of the centroid types.
struct PQTable{N,T}
    # Centroids - one for each partition of the data space
    centroids::Matrix{T}
end

function PQTable{N}(centroids::Matrix{T}) where {N,V,T}
    if size(centroids, 2) != N
        err = ArgumentError("""
        Centroid matrix for PQTable with $N partitions must have $N columns.
        Instead, it has $(size(centroids, 2)) columns!
        """)
        throw(err)
    end
    return PQTable{N,T}(centroids)
end

# TODO: Work on making inference work better here.
# Currently, not everything is statically typed.
function PQTable{N}(data::Vector{Euclidean{M,T}}, num_centroids::Integer) where {N, M, T}
    # How many element are in each partition?
    # `_valdiv` will error if `N` does not divide the data size.
    partition_size = exact_div(M, N)
    centroids = ntuple(Val(N)) do i
        _data = getindex.(cast.(Euclidean{partition_size,T}, data), i)
        centroids = choose_centroids(
            _data,
            num_centroids;
            num_iterations = 10,
            oversample = 10
        )
        return lloyds(centroids, _data; max_iterations = 100, tol = 1E-5)
    end

    centroids = reduce(hcat, centroids)
    return PQTable{N}(centroids)
end
encoded_length(::PQTable{N}) where {N} = N

refpointer(::Type{T}, x::Ref{U}) where {T,U} = Ptr{T}(Base.unsafe_convert(Ptr{U}, x))
refpointer(x::Ref{T}) where {T} = refpointer(T, x)

# Encoding
# We use unsafe version dealing directly with pointers for efficiency's sake.
# Directly moving around big tuples is expensive and it's hard to iteratively build large
# tuples.
#
# So instead, we allocate space for the tuple ahead of time and periodically update
# the correct regions.
#
# It's not the cleanest thing in the world, but should work.
encode(encoder, x) = encode(UInt64, encoder, x)
function encode(::Type{U}, encoder, x) where {U}
    ref = Ref(ntuple(_ -> zero(U), encoded_length(encoder)))

    # Keep the compiler from trying to optimize away `ref`.
    # This is likely not needed, but can't hurt as this is a generic method that is not
    # likely to be used in performance critical code paths.
    GC.@preserve ref begin
        unsafe_encode!(refpointer(U, ref), encoder, x)
    end
    return ref[]
end

unsafe_encode!(ptr::Ptr, encoder, x) = _unsafe_encode_fallback!(ptr, encoder, x)
function _unsafe_encode_fallback!(
    ptr::Ptr{U},
    table::PQTable{N,T},
    x
) where {U <: Unsigned, N, T}
    @unpack centroids = table

    dx = cast(Euclidean{length(T),eltype(x)}, x)
    for i in 1:N
        min_index = 0
        min_distance = typemax(Float32)

        for (index, centroid) in enumerate(view(centroids, :, i))
            new_distance = distance(dx[i], centroid)
            if new_distance < min_distance
                min_distance = new_distance
                min_index = index
            end
        end

        val = U(min_index - 1)
        unsafe_store!(ptr, val)
        ptr += sizeof(val)
    end
end

# Compute distance for a vector of data points.
function encode(::Type{U}, encoder, x::AbstractVector; allocator = stdallocator) where {U}
    N = encoded_length(encoder)
    data = allocator(NTuple{N,U}, length(x))
    dynamic_thread(eachindex(data)) do i
        unsafe_encode!(Ptr{U}(pointer(data, i)), encoder, x[i])
    end
    return data
end

# Assymetric Distance Computation
# Generic entry point.
#function (table::PQTable{N,T})(a, b::NTuple{N, <:Integer}) where {N,T,V}
#    return compute_distance_fallback(table, a, b)
#end

# Default distance computations
compute_distance_fallback(table, a, b) = distance(a, decode(table, b))

function decode(table::PQTable, b::NTuple{N, T}) where {N,T}
    @unpack centroids = table
    merge(ntuple(i -> @inbounds(centroids[Int(b[i]) + one(Int), i]), Val(N)))
end

#####
##### Convert a graph into a PQ-form
#####

# Has the same structure as the adjacency list of a graph, but the entries are NTuples
# that point to centroids
struct PQGraph{T <: Unsigned, N}
    raw::Vector{NTuple{N,T}}
    spans::Vector{Span{NTuple{N,T}}}

    # -- very specific inner constructor
    function PQGraph{T, N}(
        raw::Vector{NTuple{N,T}},
        spans::Vector{Span{NTuple{N,T}}},
    ) where {T <: Unsigned, N}
        return new{T,N}(raw, spans)
    end
end

# Access the adjacency list
Base.getindex(graph::PQGraph, i) = graph.spans[i]

# Specialize construction of the PQGraph for graphs implemented using the
# `DenseAdjacencyList`.
#
# Could implement a generic fallback, but in practice it ends up being so slow that it's
# not particularly useful anyways.
function PQGraph{T}(
    encoder,
    graph::UniDirectedGraph{<:Any, <:DenseAdjacencyList},
    data::AbstractVector;
    allocator = stdallocator,
) where {T,U}
    # Pre-allocate destination
    N = encoded_length(encoder)
    encoded_data = encode(T, encoder, data; allocator = allocator)
    raw = allocator(NTuple{N,T}, LightGraphs.ne(graph))

    # Since this method is specialized on using a DenseAdjacencyList, we can take
    # advantage of the underlying representation of the adjacency list being a contiguous
    # vector to parallelize the edge translation.
    @unpack storage = fadj(graph)
    meter = ProgressMeter.Progress(length(storage), 1, "Translating Adjacency Lists ... ")
    batchsize = 16384
    dynamic_thread(batched(eachindex(storage), batchsize)) do range
        for i in range
            @inbounds raw[i] = encoded_data[storage[i]]
        end
        ProgressMeter.next!(meter; step = batchsize)
    end
    ProgressMeter.finish!(meter)

    # Now that the raw storage has been computed, we just need to construct the
    # Spans encoding each region.
    accumulator = 1
    spans = map(LightGraphs.vertices(graph)) do v
        num_neighbors = length(LightGraphs.outneighbors(graph, v))
        span = Span(pointer(raw, accumulator), num_neighbors)
        accumulator += num_neighbors
        return span
    end
    return PQGraph{T,N}(raw, spans)
end

