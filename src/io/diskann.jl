# DiskANN compatible loaders
#
# TODO: Think about padding to cache line sizes if needed.
struct DiskANNLoader end

#####
##### Data Loader
#####

function load_data(::Type{T}, loader::DiskANNLoader, path::AbstractString) where {T}
    return open(path; read = true) do io
        load_data(T, loader, io)
    end
end

function load_data(::Type{T}, loader::DiskANNLoader, io::IO) where {T}
    num_points = read(io, Int32)
    dim = read(io, Int32)
    # Specialze loader on dimension of the data
    #
    # NB. For some reason, we actually need an *instance* of the decoding type, rather
    # than just the type.
    #
    # For some reason, if we just pass the type `Euclidean{dim,T}`, it doesn't get treated
    # as an `isbitstype` and the inner function explodes.
    return _load_data(Euclidean{dim,T}(), loader, io, num_points)
end

@noinline function _load_data(::E, ::DiskANNLoader, io::IO, num_points) where {E}
    dest = Vector{E}(undef, num_points)
    read!(io, dest)
    return dest
end

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
            LightGraphs.add_edge!(graph, u, v)
        end
        v += 1

        if verbose && iszero(mod(v, 100000))
            print(".")
        end
    end
    verbose && println()
    return graph
end
