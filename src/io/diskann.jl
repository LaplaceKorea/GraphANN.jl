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
struct DiskANN <: AbstractIOFormat end

#####
##### Graph Loader
#####

"""
    load_graph(GraphANN.DiskANN(), path::AbstractString, max_vertices::Integer)

Load a graph saved by the DiskANN C++ code into memory as a `UniDirectedGraph`.
"""
function load_graph(loader::DiskANN, path::AbstractString, max_vertices; kw...)
    return open(path; read = true) do io
        load_graph(loader, io, max_vertices; kw...)
    end
end

function load_graph(
    ::DiskANN, io::IO, max_vertices; verbose = true, allocator = stdallocator
)
    expected_file_size = read(io, UInt64)
    _width = read(io, Cuint)
    _ep = read(io, Cuint)

    # Use a buffer to temporarily store edge information.
    buffer = UInt32[]
    graph = UniDirectedGraph{UInt32,FlatAdjacencyList{UInt32}}(
        max_vertices, _width; allocator = allocator
    )

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
    save(path::Union{AbstractString, IO}, index::DiskANNIndex)

Save the graph in a binary format compatible with the DiskANN C++ code.
Return the number of bytes written.
"""
save(path::AbstractString, index::DiskANNIndex) = save(DiskANN(), path, index)

function save(::DiskANN, io::IO, index::DiskANNIndex)
    @unpack graph, data, startnode = index
    entry_point = startnode.index

    # Compute how large the file will be when it's created.
    filesize =
        sizeof(UInt64) +
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

function save_bin(
    ::DiskANN,
    io::IO,
    data::AbstractMatrix,
    num_points = size(data, 2),
    point_dim = size(data, 1),
)
    write(io, Cuint(num_points))
    write(io, Cuint(point_dim))
    return write(io, data)
end

"""
    load_bin(::GraphANN.DiskANN, ::Type{T}, path::AbstractString; [allocator], [groundtruth])

Load DiskANN generated binary data located at `path` with data type `T` into memory
allocated by `allocator`.

# Example
```
julia> using GraphANN

julia> data = GraphANN.load_bin(GraphANN.DiskANN(), GraphANN.SVector{128,Float32}, joinpath(GraphANN.DATADIR, "diskann", "siftsmall_query.bin)
100-element Vector{StaticArrays.SVector{128, Float32}}:
...
```

# Keyword Arguments
* `allocator` - Source of Dynamic memory allocation. Default: `stdallocator`.
* `groundtruth::Bool` - Indicate if a groundtruth dataset is being loaded. If so, then
"""
function load_bin(
    diskann::DiskANN,
    ::Type{T},
    path::AbstractString;
    allocator = stdallocator,
    groundtruth = false,
) where {T}
    return open(path) do io
        load_bin(diskann, T, io; allocator, groundtruth)
    end
end

function load_bin(
    diskann::DiskANN,
    ::Type{SVector{N,T}},
    io::IO;
    allocator = stdallocator,
    groundtruth = false,
) where {T,N}
    num_points = read(io, Cuint)
    point_dim = read(io, Cuint)
    if N != point_dim
        msg = """
        Point dimension mismatch. Expected dataset with $N dimensions. Instead, found $point_dim
        """
        throw(ArgumentError(msg))
    end
    data = allocator(SVector{N,T}, num_points)
    read!(io, data)
    if groundtruth
        # Increment each entry by 1 to account for Julia's index-1 system.
        for i in eachindex(data)
            @inbounds data[i] = data[i] .+ one(T)
        end
    end
    return data
end

function load_bin(
    diskann::DiskANN, ::Type{T}, io::IO; allocator = stdallocator, groundtruth = false
) where {T}
    num_points = read(io, Cuint)
    point_dim = read(io, Cuint)
    data = allocator(T, point_dim, num_points)
    read!(io, data)
    if groundtruth
        data .+= one(T)
    end
    return data
end
