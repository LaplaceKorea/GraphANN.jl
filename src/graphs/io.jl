#####
##### Serialization
#####

function save(file::AbstractString, x; kw...)
    open(file; write = true) do io
        save(io, x; kw...)
    end
    return nothing
end

save(io::IO, meta::MetaGraph{<:UniDirectedGraph}; kw...) = save(io, meta.graph; kw...)
function save(
    io::IO,
    g::UniDirectedGraph{T};
    buffersize = div(2_000_000_000, sizeof(T))
) where {T}
    write_header(io, g)

    # Write number of neighbors
    write(io, T.(LightGraphs.outdegree(g)))

    # Write the adjacency lists themselves.
    # Queue up the adjacency lists into large chunks to try to get slightly better
    # bandwidth.
    buffer = T[]
    sizehint!(buffer, buffersize)
    last_vertex = LightGraphs.nv(g)

    ProgressMeter.@showprogress 1 for v in LightGraphs.vertices(g)
        append!(buffer, LightGraphs.outneighbors(g, v))
        if length(buffer) >= buffersize || v == last_vertex
            write(io, buffer)
            empty!(buffer)
        end
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

header_length() = 4

load(file::AbstractString; kw...) = load(DefaultAdjacencyList{UInt32}, file; kw...)
function load(::Type{T}, file::AbstractString; kw...) where {T}
    return open(io -> load(T, io; kw...), file)
end

function load(::Type{DefaultAdjacencyList{T}}, io::IO) where {T}
    # Read the header
    @unpack elsize, nv, ne, max_degree = read_header(io)
    @assert elsize == sizeof(T)

    # Allocate destination array.
    outdegrees = Vector{T}(undef, nv)
    read!(io, outdegrees)

    adj = DefaultAdjacencyList{T}()
    ProgressMeter.@showprogress 1 for v in 1:nv
        buffer = T[]
        resize!(buffer, outdegrees[v])
        read!(io, buffer)
        push!(adj, buffer)
    end

    eof(io) || error("There seems to be more data left on the file!")
    return UniDirectedGraph{T}(adj)
end

function load(
    ::Type{FlatAdjacencyList{T}},
    io::IO;
    pad_to = nothing,
    allocator = stdallocator
) where {T}
    @unpack elsize, nv, ne, max_degree = read_header(io)
    @assert elsize == sizeof(T)

    # If padding is requested, determine the number of elements of type T corresponds to
    # the requested padding.
    if pad_to !== nothing
        num_elements = div(pad_to, sizeof(T))
        if num_elements * sizeof(T) != pad_to
            throw(ArgumentError(
                "Requested padding bytes of $pad_to is not evenly divisible by elements of type $T"
            ))
        end
        max_degree = cdiv(max_degree, num_elements) * num_elements
    end

    outdegrees = Vector{T}(undef, nv)
    read!(io, outdegrees)

    # Allocate destination array.
    g = UniDirectedGraph{T}(FlatAdjacencyList{T}(nv, max_degree; allocator))
    buffer = T[]
    ProgressMeter.@showprogress 1 for v in 1:nv
        resize!(buffer, outdegrees[v])
        read!(io, buffer)
        copyto!(g, v, buffer)
    end

    # Did we exhaust the whole file?
    eof(io) || error("There seems to be more data left on the file!")
    return g
end

function load(
    ::Type{DenseAdjacencyList{T}},
    io::IO;
    allocator = stdallocator,
) where {T}
    @unpack elsize, nv, ne, max_degree = read_header(io)
    @assert elsize == sizeof(T)

    # Load how long each adjacency list is.
    offsets = Vector{Int}(undef, nv + 1)
    offsets[1] = 1
    ProgressMeter.@showprogress 1 for i in 1:nv
        @inbounds offsets[i+1] = offsets[i] + read(io, T)
    end

    # Preallocate the storage and load everything into memory.
    A = allocator(T, ne)
    read!(io, A)
    eof(io) || error("There seems to be more room in the file!")

    adj = DenseAdjacencyList{T}(A, offsets)
    return UniDirectedGraph{T}(adj)
end
