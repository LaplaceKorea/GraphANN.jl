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
# V - The element type of the wrapped dataset.
# T - The element type of the centroid types.
struct PQTable{N,V,T}
    # The data that we're quantizing
    data::Vector{V}

    # Centroids - one for each partition of the data space
    centroids::Matrix{T}
end

function PQTable{N}(data::Vector{V}, centroids::Matrix{T}) where {N,V,T}
    return PQTable{N,V,T}(data, centroids)
end

# TODO: Work on making inference work better here.
# Currently, not everything is statically typed.
function PQTable{N}(data::Vector{Euclidean{M,T}}, num_centroids::Integer) where {N, M, T}
    # How many element are in each partition?
    # `_valdiv` will error if `N` does not divide the data size.
    partition_size = exact_div(M, N)
    centroids = ntuple(Val(N)) do i
        _data = getindex.(deconstruct.(Euclidean{partition_size,T}, data), i)
        centroids = choose_centroids(
            _data,
            num_centroids;
            num_iterations = 10,
            oversample = 5
        )
        return lloyds(centroids, _data; max_iterations = 100)
    end

    centroids = reduce(hcat, centroids)
    return PQTable{N}(data, centroids)
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
    table::PQTable{N,V,T},
    x::V
) where {U <: Unsigned, N, V <: Euclidean, T}
    @unpack centroids = table

    dx = deconstruct(Euclidean{length(T),eltype(x)}, x)
    for i in 1:N
        current_minimum = CurrentMinimum()
        for (index, centroid) in enumerate(view(centroids, :, i))
            candidate = CurrentMinimum(distance(dx[i], centroid), index)
            if candidate < current_minimum
                current_minimum = candidate
            end
        end

        val = U(current_minimum.index - 1)
        unsafe_store!(ptr, val)
        ptr += sizeof(val)
    end
end

# Compute distance for a vector of data points.
function encode(::Type{U}, encoder, x::AbstractVector) where {U}
    data = Vector{NTuple{encoded_length(encoder),U}}(undef, length(x))
    dynamic_thread(eachindex(data)) do i
        unsafe_encode!(Ptr{U}(pointer(data, i)), encoder, x[i])
    end
    return data
end

# Assymetric Distance Computation
# Generic entry point.
function (table::PQTable{N,V,T})(a::V, b::NTuple{N, <:Integer}) where {N,T,V}
    return compute_distance_fallback(table, a, b)
end

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
    pqtable::PQTable{N},
    graph::UniDirectedGraph{<:Any, <:DenseAdjacencyList};
    allocator = stdallocator,
    code_type::Type{U} = UInt8,
    encoder = pqtable,
) where {N,T,U}
    # Pre-allocate destination
    raw = allocator(NTuple{N,T}, LightGraphs.ne(graph))

    # Since this method is specialized on using a DenseAdjacencyList, we can take
    # advantage of the underlying representation of the adjacency list being a contiguous
    # vector to parallelize the edge translation.
    @unpack storage = fadj(graph)
    @unpack data = pqtable
    dynamic_thread(eachindex(storage), 32) do i
        v = storage[i]
        unsafe_encode!(convert(Ptr{U}, pointer(raw, i)), encoder, data[v])
    end

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

