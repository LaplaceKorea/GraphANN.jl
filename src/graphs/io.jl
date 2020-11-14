#####
##### Serialization
#####

function save(file::AbstractString, g::UniDirectedGraph)
    open(file; write = true) do io
        save(io, g)
    end
    return nothing
end

function save(io::IO, g::UniDirectedGraph{T}) where {T}
    # Save some stats about the graph in a header
    write(io, Int64(sizeof(T)))
    write(io, Int64(LightGraphs.nv(g)))
    write(io, Int64(maximum(LightGraphs.outdegree(g))))

    # Now, serialize the adjacency list
    ProgressMeter.@showprogress 1 for v in LightGraphs.vertices(g)
        neighbors = LightGraphs.outneighbors(g, v)
        write(io, T(length(neighbors)))
        write(io, neighbors)
    end
    return nothing
end

load(file::AbstractString) = open(io -> load(DefaultAdjacencyList{UInt32}, io), file)

function load(::Type{FlatAdjacencyList{T}}, io::IO) where {T}
    # Read the header
    elsize = read(io, Int64)
    nv = read(io, Int64)
    max_degree = read(io, Int64)

    @assert elsize == sizeof(T)

    # Allocate destination array.
    g = UniDirectedGraph{T}(FlatAdjacencyList{T}(nv, max_degree))
    v = 1
    buffer = T[]

    progress_meter = ProgressMeter.Progress(nv, 1)

    while (v <= nv && !eof(io))
        num_neighbors = read(io, T)
        resize!(buffer, num_neighbors)
        read!(io, buffer)
        copyto!(g, v, buffer)

        # Loop footer (obviously)
        ProgressMeter.next!(progress_meter)
        v += 1
    end

    # Did we exhaust the whole file?
    v != (nv + 1) && error("Finished reading file before all vertices were discovered!")
    eof(io) || error("There seems to be more data left on the file!")
    return g
end

function load(::Type{DefaultAdjacencyList{T}}, io::IO) where {T}
    # Read the header
    elsize = read(io, Int64)
    nv = read(io, Int64)
    max_degree = read(io, Int64)

    @assert elsize == sizeof(T)

    # Allocate destination array.
    adj = DefaultAdjacencyList{T}()
    v = 1

    progress_meter = ProgressMeter.Progress(nv, 1)

    while (v <= nv && !eof(io))
        buffer = T[]
        num_neighbors = read(io, T)
        resize!(buffer, num_neighbors)
        read!(io, buffer)
        push!(adj, buffer)

        # Loop footer (obviously)
        ProgressMeter.next!(progress_meter)
        v += 1
    end

    # Did we exhaust the whole file?
    v != (nv + 1) && error("Finished reading file before all vertices were discovered!")
    eof(io) || error("There seems to be more data left on the file!")
    return UniDirectedGraph{T}(adj)
end
