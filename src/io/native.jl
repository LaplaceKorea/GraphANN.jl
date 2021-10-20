#####
##### Graph Serialization
#####

function save(
    ::Native, io::IO, g::UniDirectedGraph{T}; buffersize = div(2_000_000_000, sizeof(T))
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
    return write(io, Int64(maximum(LightGraphs.outdegree(g))))
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

"""
    load(::Type{T}, io::Union{String, IO}; kw...) -> UniDirectedGraph

Load a graph in canonical GraphANN binary form into memory using an adjacency list
representation given by `T`. Acceptable types `T` and their corresponding keywords are
given below. **Note**: The parameter `U` given below must be either `UInt32` or `UInt64`
and match what was used to save the graph originally.

* `GraphANN.DefaultAdjacencyList{U}` - Use the default representation for an adjacency list
    (vector of vectors). No keyword arguments.

* `GraphANN.FlatAdjacencyList{U}` - Use the flat adjacency list representation. Keywords:
    - `pad_to::Integer`: Try to pad each column of the flat matrix to this many bytes.
    - `allocator`: Allocator for memory. Default: [`stdallocator`](@ref).

* `GraphANN.DenseAdjacencyList{U}` - Use the dense adjacency list representation. Keywords:
    - `allocator`: Allocator for memory. Default: [`stdallocator`](@ref).
"""
function load(
    ::Type{T}, file::AbstractString; kw...
) where {T<:_Graphs.AbstractAdjacencyList}
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
    ::Type{U}, io::IO; pad_to = nothing, allocator = stdallocator
) where {T,U<:AbstractFlatAdjacencyList{T}}
    @unpack elsize, nv, ne, max_degree = read_header(io)
    @assert elsize == sizeof(T)

    # If padding is requested, determine the number of elements of type T corresponds to
    # the requested padding.
    if pad_to !== nothing
        num_elements = div(pad_to, sizeof(T))
        if num_elements * sizeof(T) != pad_to
            throw(
                ArgumentError(
                    "Requested padding bytes of $pad_to is not evenly divisible by elements of type $T",
                ),
            )
        end
        max_degree = cdiv(max_degree, num_elements) * num_elements
    end

    outdegrees = Vector{T}(undef, nv)
    read!(io, outdegrees)

    # Allocate destination array.
    g = UniDirectedGraph{T}(U(nv, max_degree; allocator))
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

function load(::Type{DenseAdjacencyList{T}}, io::IO; allocator = stdallocator) where {T}
    @unpack elsize, nv, ne, max_degree = read_header(io)
    @assert elsize == sizeof(T)

    # Load how long each adjacency list is.
    offsets = Vector{Int}(undef, nv + 1)
    offsets[1] = 1
    ProgressMeter.@showprogress 1 for i in 1:nv
        @inbounds offsets[i + 1] = offsets[i] + read(io, T)
    end

    # Preallocate the storage and load everything into memory.
    A = allocator(T, ne)
    read!(io, A)
    eof(io) || error("There seems to be more room in the file!")

    adj = DenseAdjacencyList{T}(A, offsets)
    return UniDirectedGraph{T}(adj)
end

#####
##### Binary Serialization
#####

"""
    save_bin(io::Union{AbstractString, IO}, v)

Save `v` to `io` in canonical binary format such that `load_bin(io, typeof(v))` works.
"""
save_bin(path::AbstractString, v::AbstractVector) = open(io -> save_bin(io, v))
save_bin(io::IO, v::AbstractVector) = write(io, v)

# Store data structures a memory-mappable files.
# If these files live in PM, then it makes statup cost SIGNIFICANTLY lower.
"""
    load_bin(io::Union{AbstractString, IO}, ::Type{Vector{T}})

Memory map `io` as a vector with eltype `T`.
"""
function load_bin(path::AbstractString, ::Type{Vector{T}}; write = true, kw...) where {T}
    return open(path; read = true, write = write) do io
        load_bin(io, Vector{T}; kw...)
    end
end

load_bin(io::IO, ::Type{Vector{T}}) where {T} = Mmap.mmap(io, Vector{T})

# Graphs
"""
    save_bin(dir::AbstractString, graph::UniDirectedGraph{T, DenseAdjacencyList{T}})

Save graph as multiple files in `dir`.

## Implementation Details

The CSR offsets and dense adjacency list are saved as canonical binary files in
`joinpath(dir, "offsets.bin")` and `joinpath(dir, "neighbors.bin")` respectively.
"""
function save_bin(
    dir::AbstractString, graph::UniDirectedGraph{T,_Graphs.DenseAdjacencyList{T}}
) where {T}
    !isdir(dir) && mkdir(dir)

    # Save Offsets
    bytes_written = 0
    open(joinpath(dir, "offsets.bin"); write = true) do io
        bytes_written += write(io, _Graphs.fadj(graph).offsets)
    end

    # Save neighbors
    open(joinpath(dir, "neighbors.bin"); write = true) do io
        bytes_written += write(io, _Graphs.fadj(graph).storage)
    end
    return bytes_written
end

# Convenience function
function load_bin(path::AbstractString, ::Type{A}; kw...) where {T,A<:AbstractAdjacencyList{T}}
    return load_bin(path, UniDirectedGraph{T,A}; kw...)
end

"""
    load_bin(dir::AbstractString, GraphANN.DenseAdjacencyList{T})

Perform a fast load of a graph using the [`DenseAdjacencyList`](@ref) that was saved as
multiple files in `dir` using the
[`save_bin`](@ref save_bin(::AbstractString, ::UniDirectedGraph{T, _Graphs.DenseAdjacencyList{T}}) where {T}) function.

## Implementation Detail

The CSR offsets are copied into DRAM while the dense adjacency list is memory mapped.
"""
function load_bin(
    dir::AbstractString, ::Type{UniDirectedGraph{T,_Graphs.DenseAdjacencyList{T}}}
) where {T}
    # Load offsets
    # Copy the MMap in case the original file lives in PM.
    offsets = copy(Mmap.mmap(joinpath(dir, "offsets.bin"), Vector{Int64}))
    storage = Mmap.mmap(joinpath(dir, "neighbors.bin"), Vector{T})

    return UniDirectedGraph{T}(_Graphs.DenseAdjacencyList{T}(storage, offsets))
end

function load_bin(
    path::AbstractString,
    ::Type{UniDirectedGraph{T,_Graphs.SuperFlatAdjacencyList{T}}};
    writable = false,
) where {T}
    storage = open(path; read = true, write = writable) do io
        elsize, nv, _, maxdegree = read_header(io)
        if elsize != sizeof(T)
            msg = """
            Trying to load a graph that was saved with an elements sized $elsize bytes.
            The passed data type was $T which has a size of $(sizeof(T)) bytes.
            """
            error(msg)
        end
        # Compute the offset consumed by the header.
        # Memory map from this offset on.
        offset = elsize * (maxdegree + 1)
        return Mmap.mmap(io, Matrix{T}, (maxdegree + 1, nv), offset)
    end
    return UniDirectedGraph{T}(SuperFlatAdjacencyList{T}(storage))
end

#####
##### Save to Mmap
#####

"""
    save_as_superflat(path, graph; [batchsize]) -> newgraph

Save `graph` in a superflat format to `path`.
This format is quickly loadable via
[`load_bin(path, GraphANN.UniDirectedGraph{T,GraphANN.SuperFlatAdjacencyList{T}})`]

NOTE: The graph should be loaded through the `load_bin` function instead of directly
memory mapping because the saved format contains metadata regarding the number of
vertices and max degree.
"""
function save_as_superflat(
    path::AbstractString, graph::UniDirectedGraph{T}; batchsize = 2048
) where {T}
    # Compute how much space we are going to need
    max_degree = maximum(LightGraphs.outdegree(graph))
    elements_per_vertex = max_degree + 1
    storage = open(path; read = true, write = true, create = true) do io
        write_header(io, graph)
        offset = sizeof(T) * elements_per_vertex
        return Mmap.mmap(
            io, Matrix{T}, (elements_per_vertex, LightGraphs.nv(graph)), offset
        )
    end
    newgraph = UniDirectedGraph{T}(SuperFlatAdjacencyList{T}(storage))

    iter = batched(Base.OneTo(LightGraphs.nv(graph)), batchsize)
    progress = ProgressMeter.Progress(length(iter), 1)
    dynamic_thread(iter) do range
        for v in range
            copyto!(newgraph, v, LightGraphs.outneighbors(graph, v))
        end
        ProgressMeter.next!(progress)
    end
    return newgraph
end
