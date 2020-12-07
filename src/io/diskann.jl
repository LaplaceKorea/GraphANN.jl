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


function save_graph(loader::DiskANNLoader, path::AbstractString, meta::MetaGraph)
    return open(path; write = true) do io
        save_graph(loader, io, meta)
    end
end

function save_graph(::DiskANNLoader, io::IO, meta::MetaGraph)
    @unpack graph, data = meta

    write(io, UInt64(LightGraphs.nv(graph)))
    write(io, Cuint(maximum(LightGraphs.outdegree(graph))))
    write(io, Cuint(medioid(data) - 1))
    for v in LightGraphs.vertices(graph)
        neighbors = LightGraphs.outneighbors(graph, v)j
        write(io, Cuint(length(neighbors)))
        for u in neighbors
            write(io, Cuint(u-1))
        end
    end
end
