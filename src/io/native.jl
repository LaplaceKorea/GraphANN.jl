# Dataset
function load_bin(path::AbstractString, ::Type{Vector{T}}; write = false, kw...) where {T}
    return open(path; read = true, write = write) do io
        load_bin(io, Vector{T}; kw...)
    end
end

load_bin(io::IO, ::Type{Vector{T}}) where {T} = Mmap.mmap(io, Vector{T})

# Graphs
function save_bin(
    dir::AbstractString,
    graph::UniDirectedGraph{T, _Graphs.DenseAdjacencyList{T}},
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

function load_bin(
    dir::AbstractString,
    ::Type{UniDirectedGraph{T, _Graphs.DenseAdjacencyList{T}}},
) where {T}
    # Load offsets
    # Copy the MMap in case the original file lives in PM.
    offsets = copy(Mmap.mmap(joinpath(dir, "offsets.bin"), Vector{Int64}))
    storage = Mmap.mmap(joinpath(dir, "neighbors.bin"), Vector{T})

    return UniDirectedGraph{T}(_Graphs.DenseAdjacencyList{T}(storage, offsets))
end


