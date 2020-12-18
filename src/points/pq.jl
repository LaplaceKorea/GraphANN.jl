#####
##### Product Quantize a dataset
#####

function _valdiv(A, B)
    div, rem = divrem(A, B)
    if iszero(rem)
        return div
    else
        return error("Data length $A is not divisible by the number of partitions $B")
    end
end

struct PQTable{N,V,T}
    # The data that we're quantizing
    data::Vector{V}

    # Centroids - one for each partition of the data space
    centroids::NTuple{N, Vector{T}}
end

# TODO: Work on making inference work better here.
# Currently, not everything is statically typed.
function PQTable{N}(data::Vector{Euclidean{M,T}}, num_centroids) where {N, M, T}
    # How many element are in each partition?
    # `_valdiv` will error if `N` does not divide the data size.
    partition_size = _valdiv(M, N)

    centroids = ntuple(Val(N)) do i
        _data = getindex.(deconstruct.(Euclidean{partition_size,T}, data), i)
        centroids = choose_centroids(
            _data,
            num_centroids;
            num_iterations = 10,
            oversample = 5
        )
        lloyds!(centroids, _data; max_iterations = 100)
        return centroids
    end

    return PQTable(data, centroids)
end

# Encoding
encode(table::PQTable, x) = encode(UInt64, table, x)
function encode(::Type{U}, table::PQTable{N,V,T}, x::V) where {U <: Unsigned, N,V,T}
    @unpack centroids = table

    dx = deconstruct(T, x)
    return ntuple(Val(N)) do i
        current_minimum = CurrentMinimum()
        for (index, centroid) in enumerate(centroids[i])
            candidate = CurrentMinimum(distance(dx[i], centroid), index)
            if candidate < current_minimum
                current_minimum = candidate
            end
        end
        return U(current_minimum.index - 1)
    end
end

# Assymetric Distance Computation
function (table::PQTable{N,V,T})(a::V, b::NTuple{N, <:Integer}) where {N,T,V}
    distance(a, decode(table, b))
end

function decode(table::PQTable, b::NTuple{N, T}) where {N,T}
    @unpack centroids = table
    merge(ntuple(i -> @inbounds(centroids[i][b[i] + one(T)]), Val(N)))
end

function _query(table::PQTable, queries::AbstractVector)
    results = Array{Int64,1}(undef, num_neighbors)
    dynamic_thread(eachindex(queries)) do query_index
        query = queries[query_index]
        current_minimum = CurrentMinimum()
        for (index, x) in enumerate(centroids[i])
            candidate = CurrentMinimum(distance(dx[i], centroid), index)
            if candidate < current_minimum
                current_minimum = candidate
            end
        end
        return U(current_minimum.index - 1)

    end
end

#####
##### Convert a graph into a PQ-form
#####

# Has the same structure as the adjacency list of a graph, but the entries are NTuples
# that point to centroids
struct PQGraph{N,T <: Unsigned}
    raw::Vector{NTuple{N,T}}
    spans::Vector{Span{NTuple{N,T}}}

    # -- very specific inner constructor
    function PQGraph{N, T}(
        raw::Vector{NTuple{N,T}},
        spans::Vector{Span{NTuple{N,T}}},
    ) where {N, T <: Unsigned}
        return new{N,T}(raw, spans)
    end
end

# Access the adjacency list
Base.getindex(graph::PQGraph, i) = graph.spans[i]

function PQGraph{N,T}(
    pqtable::PQTable{N},
    graph::UniDirectedGraph{<:Any, <:DenseAdjacencyList};
    allocator = stdallocator
) where {N,T}
    # Pre-allocate destination
    raw = allocator(NTuple{N,T}, LightGraphs.ne(graph))

    # Since this method is specialized on using a DenseAdjacencyList, we can take
    # advantage of the underlying representation of the adjacency list being a contiguous
    # vector to parallelize the edge translation.
    @unpack storage = fadj(graph)
    @unpack data = pqtable
    dynamic_thread(eachindex(storage), 32) do i
        v = storage[i]
        raw[i] = encode(pqtable, data[v])
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
    return PQGraph{N,T}(raw, spans)
end

# Geneic fallback
function PQGraph{N,T}(
    pqtable::PQTable{N},
    graph::UniDirectedGraph{<:Any};
    allocator = stdallocator
) where {N,T}
    # Pre-allocate destination
    raw = allocator(NTuple{N,T}, LightGraphs.ne(graph))

    # Since this method is specialized on using a DenseAdjacencyList, we can take
    # advantage of the underlying representation of the adjacency list being a contiguous
    # vector to parallelize the edge translation.
    @unpack data = pqtable
    index = 1
    ProgressMeter.@showprogress 1 for v in LightGraphs.vertices(graph)
        for u in LightGraphs.outneighbors(graph, v)
            raw[index] = encode(pqtable, data[u])
        end
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
    return Span(raw, spans)
end


