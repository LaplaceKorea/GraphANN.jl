# Graph Loading
function loadgraph(path)
    return GraphANN.load_bin(
        path, GraphANN.SuperFlatAdjacencyList{UInt32}; writable = false
    )
end

function loaddata(path, ::Type{T}, dim::Integer) where {T}
    return GraphANN.load_bin(path, Vector{StaticArrays.SVector{dim,T}})
end

function loadindex(dir, ::Type{T}, dim, metric) where {T}
    graph = GraphANN.load_bin(
        joinpath(dir, "graph.bin"),
        GraphANN.SuperFlatAdjacencyList{UInt32};
        writable = false,
    )

    data = GraphANN.load_bin(
        joinpath(dir, "data.bin"),
        Vector{StaticArrays.SVector{dim,T}},
    )

    index = GraphANN.DiskANNIndex(graph, data, metric)
    return index
end
