# DiskANN compatible loaders
struct DiskANNLoader end

#####
##### Graph Loader
#####

function load_graph(loader::DiskANNLoader, path::AbstractString, max_vertices; kw...)
    return open(path; read = true) do io
        load_graph(loader, io, max_vertices; kw...)
    end
end

function load_graph(::DiskANNLoader, io::IO, max_vertices; verbose = true)
    # Load some of the header information (I'm not sure that `_width` and `_ep` are).
    expected_file_size = read(io, UInt64)
    _width = read(io, Cuint)
    _ep = read(io, Cuint)

    # Use a buffer to temporarily store edge information.
    buffer = UInt32[]
    graph = LightGraphs.SimpleDiGraph{UInt32}(max_vertices)

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
