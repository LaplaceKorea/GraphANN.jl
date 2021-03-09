# DiskANN compatible loaders
"""
Singleton type to resolve ambiguities.

## Constructor

```jldoctest; output = false
GraphANN.DiskANN()

# output

GraphANN._IO.DiskANN()
```
"""
struct DiskANN end

#####
##### Graph Loader
#####

"""
    load_graph(GraphANN._IO.DiskANN(), path::AbstractString, max_vertices::Integer)

Load a graph saved by the DiskANN C++ code into memory as a `UniDirectedGraph`.
"""
function load_graph(loader::DiskANN, path::AbstractString, max_vertices; kw...)
    return open(path; read = true) do io
        load_graph(loader, io, max_vertices; kw...)
    end
end

function load_graph(::DiskANN, io::IO, max_vertices; verbose = true)
    expected_file_size = read(io, UInt64)
    _width = read(io, Cuint)
    _ep = read(io, Cuint)

    # Use a buffer to temporarily store edge information.
    buffer = UInt32[]
    graph = UniDirectedGraph{UInt32}(max_vertices)

    verbose && print("Loading Graph")

    # v is our current source vertex
    v = 1
    while !eof(io)
        num_edges = read(io, Cuint)
        resize!(buffer, num_edges)
        read!(io, buffer)

        # Numbers in the buffer are the destination vertices.
        for u in buffer
            # Adjust for 0 vs 1 based indexing
            LightGraphs.add_edge!(graph, v, u + 1)
        end
        v += 1

        if verbose && iszero(mod(v, 100000))
            print(".")
        end
    end
    verbose && println()
    return graph
end

"""
    save_graph(path::Union{AbstractString, IO}, index::DiskANNIndex)

Save the graph in a binary format compatible with the DiskANN C++ code.
Return the number of bytes written.
"""
function save_graph(path::AbstractString, index::DiskANNIndex)
    return open(io -> save_graph(io, index), path; write = true)
end

function save_graph(io::IO, index::DiskANNIndex)
    @unpack graph, data, startnode = index
    entry_point = startnode.index

    # Compute how large the file will be when it's created.
    filesize = sizeof(UInt64) +
        2 * sizeof(Cuint) +
        LightGraphs.nv(graph) * sizeof(Cuint) +
        LightGraphs.ne(graph) * sizeof(Cuint)

    # As a sanity check, make sure our file size computation was correct.
    bytes = 0
    bytes += write(io, UInt64(filesize))
    bytes += write(io, Cuint(maximum(LightGraphs.outdegree(graph))))
    bytes += write(io, Cuint(entry_point - 1))
    ProgressMeter.@showprogress 1 for v in LightGraphs.vertices(graph)
        neighbors = LightGraphs.outneighbors(graph, v)
        bytes += write(io, Cuint(length(neighbors)))
        bytes += write(io, Cuint.(neighbors .- 1))
    end

    if bytes != filesize
        error("Incorrect number of bytes written. Expected $filesize, wrote $bytes!")
    end
    return bytes
end

# function save_pq(diskann::DiskANN, path::AbstractString, args...; kw...)
#     return open(path; write = true) do io
#         save_pq(diskann, io, args...; kw...)
#     end
# end
#
# function save_pq(diskann::DiskANN, io::IO, centroids::AbstractMatrix{T}) where {T}
#     # DiskANN wants concatenate centroids together.
#     # This is the opposite of how we store them, so we need to take a transpose
#     save_bin(diskann, io, transpose(centroids), size(centroids, 1), length(T) * size(centroids, 2))
# end

"""
    save_bin(::DiskANN, path::AbstractString, data::AbstractMatrix)

Save `data` to `path` in a form compatible with DiskANN's `load_bin` function.
"""
function save_bin(diskann::DiskANN, path::AbstractString, data::AbstractMatrix)
    return open(path; write = true) do io
        save_bin(diskann, io, data)
    end
end

function save_bin(::DiskANN, io::IO, data::AbstractMatrix, num_points = size(data, 2), point_dim = size(data, 1))
    write(io, Cuint(num_points))
    write(io, Cuint(point_dim))
    write(io, data)
end

