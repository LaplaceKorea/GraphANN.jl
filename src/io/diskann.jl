# DiskANN compatible loaders
struct DiskANN end

#####
##### Graph Loader
#####

function load_graph(loader::DiskANN, path::AbstractString, max_vertices; kw...)
    return open(path; read = true) do io
        load_graph(loader, io, max_vertices; kw...)
    end
end

function load_graph(::DiskANN, io::IO, max_vertices; verbose = true)
    # Load some of the header information (I'm not sure that `_width` and `_ep` are).
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
    save_graph(::DiskANN, path::Union{AbstractString, IO}, meta::MetaGraph)

Save the graph in a binary format compatible with the DiskANN C++ code.
Return the number of bytes written.
"""
function save_graph(loader::DiskANN, path::AbstractString, meta::MetaGraph)
    return open(path; write = true) do io
        save_graph(loader, io, meta)
    end
end

function save_graph(::DiskANN, io::IO, meta::MetaGraph)
    @unpack graph, data = meta

    # Compute how large the file will be when it's created.
    filesize = sizeof(UInt64) +
        2 * sizeof(Cuint) +
        LightGraphs.nv(graph) * sizeof(Cuint) +
        LightGraphs.ne(graph) * sizeof(Cuint)

    # As a sanity check, make sure our file size computation was correct.
    bytes = 0
    bytes += write(io, UInt64(filesize))
    bytes += write(io, Cuint(maximum(LightGraphs.outdegree(graph))))
    bytes += write(io, Cuint(medioid(data) - 1))
    ProgressMeter.@showprogress 1 for v in LightGraphs.vertices(graph)
        neighbors = LightGraphs.outneighbors(graph, v)
        bytes += write(io, Cuint(length(neighbors)))
        for u in neighbors
            bytes += write(io, Cuint(u-1))
        end
    end

    if bytes != filesize
        error("Incorrect number of bytes written. Expected $filesize, wrote $bytes!")
    end
    return bytes
end

function save_pq(diskann::DiskANN, path::AbstractString, args...; kw...)
    return open(path; write = true) do io
        save_pq(diskann, io, args...; kw...)
    end
end

function save_pq(diskann::DiskANN, io::IO, centroids::AbstractMatrix{T}) where {T}
    # DiskANN wants concatenate centroids together.
    # This is the opposite of how we store them, so we need to take a transpose
    save_bin(diskann, io, transpose(centroids), size(centroids, 1), length(T) * size(centroids, 2))
end

function save_bin(diskann::DiskANN, path::AbstractString, args...)
    return open(path; write = true) do io
        save_bin(diskann, io, args...)
    end
end

function save_bin(
    ::DiskANN,
    io::IO,
    data::AbstractMatrix,
    num_points = size(data, 2),
    point_dim = size(data, 1)
)
    write(io, Cuint(num_points))
    write(io, Cuint(point_dim))
    write(io, data)
end

