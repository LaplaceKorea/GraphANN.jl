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
    write_header(io, g)

    # Now, serialize the adjacency list
    ProgressMeter.@showprogress 1 for v in LightGraphs.vertices(g)
        neighbors = LightGraphs.outneighbors(g, v)
        write(io, T(length(neighbors)))
        write(io, neighbors)
    end
    return nothing
end

function write_header(io::IO, g::UniDirectedGraph{T}) where {T}
    # Save some stats about the graph in a header
    write(io, Int64(sizeof(T)))
    write(io, Int64(LightGraphs.nv(g)))
    write(io, Int64(LightGraphs.ne(g)))
    write(io, Int64(maximum(LightGraphs.outdegree(g))))
end

# Return the header as a NamedTuple.
# This is so users of `read_header` can use `@unpack` on the results to obtain what
# they need.
#
# This allows us to add features to the header without changing the code below.
function read_header(io::IO)
    tup = read.((io,), ntuple(_ -> Int64, Val(4)))
    return NamedTuple{(:elsize, :nv, :ne, :max_degree)}(tup)
end

load(file::AbstractString; kw...) = load(DefaultAdjacencyList{UInt32}, file; kw...)
load(::Type{T}, file::AbstractString; kw...) where {T} = open(io -> load(T, io; kw...), file)

function load(::Type{FlatAdjacencyList{T}}, io::IO) where {T}
    @unpack elsize, nv, max_degree = read_header(io)
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
    @unpack elsize, nv, max_degree = read_header(io)
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

function load(
    ::Type{DenseAdjacencyList{T}},
    io::IO;
    allocator = stdallocator,
) where {T}
    # Read the header
    @unpack elsize, nv, ne, max_degree = read_header(io)
    @assert elsize == sizeof(T)

    # Preallocate the storage array and the Spans that are going to store the
    # lengths and neighbors
    A = allocator(T, ne)
    spans = Vector{Span{T}}()
    sizehint!(spans, nv)

    progress_meter = ProgressMeter.Progress(nv, 1)
    index = 1
    v = 1
    while (v <= nv && !eof(io))
        num_neighbors = read(io, T)
        read!(io, view(A, index:(index + num_neighbors - 1)))
        push!(spans, Span(pointer(A, index), num_neighbors))
        index += num_neighbors

        # Loop footer (obviously)
        ProgressMeter.next!(progress_meter)
        v += 1
    end

    # Did we exhaust the whole file?
    v != (nv + 1) && error("Finished reading file before all vertices were discovered!")
    eof(io) || error("There seems to be more data left on the file!")

    adj = DenseAdjacencyList{T}(A, spans)
    return UniDirectedGraph{T}(adj)
end

