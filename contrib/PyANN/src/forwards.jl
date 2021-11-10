#####
##### Load Index
#####

struct DirectMmap end
const direct_mmap = DirectMmap()

function loaddata(
    path,
    ::Type{T},
    dim::Integer;
    allocator = direct_mmap,
    diskann_format = false,
    numacopy = false,
) where {T}
    eltyp = StaticArrays.SVector{dim,T}
    _data = GraphANN.load_bin(path, Vector{eltyp}; offset = diskann_format ? 8 : 0)
    if allocator !== direct_mmap
        # Do the happy little dance when assigining copies to each NUMA node.
        if numacopy
            data = GraphANN.@numacopy allocator=allocator strict=true begin
                __allocator__(eltyp, length(_data))
            end

            for j in Base.OneTo(length(data))
                local_version = data[j]
                GraphANN.dynamic_thread(eachindex(_data, local_version), 2048) do i
                    @inbounds(local_version[i] = _data[i])
                end
            end
        else
            data = allocator(eltyp, length(_data))
            GraphANN.dynamic_thread(eachindex(_data, data), 2048) do i
                @inbounds(data[i] = _data[i])
            end
        end
    else
        data = _data
    end
    return data
end

_prefix(x::AbstractString) = x
_prefix(x::AbstractArray) = _prefix(x[1])

"""
    loadindex(dir, ::Type{T}, dim::Integer, metric; kw...) -> Index

Load an index stored in `dir`. Data points within the index's dataset should have type `T`
(where `T` is some machine native type like `Float33`, `UInt8` etc.) and length `dim` such
that the elements of the loaded dataset are a `SVector{dim,T}`. The `metric` to use for
the dataset is passed as the final argument. If `dir` is a vector of strings, than it is
assumed that each directory belongs to a copy of the graph and each will be loaded and
bundled into a `NumaAware` struct.

It is assumed that the graph will live in `joinpath(dir, "graph.bin")` and the data will
be at `joinpath(dir, "data.bin")`, though the full path for the dataset may be passed as
explicitly as a keyword argument. Furthermore, it is assumed that the graph will live
entirely in Persistent Memory and is encoded using a `GraphANN.SuperFlatAdjacencyList{UInt32}`.

Keywords
--------
* `datapath` - Full path to the dataset file. Default: `joinpath(dir, "data.bin")`
* `allocator` - The allocator to use for the dataset portion of the index. If this argument
    is `direct_mmap`, then the dataset will be directly memory mapped instead.
    Default: `direct_mmap`.
* `diskann_format::Bool` - Set to `true` is the dataset is in the DiskANN binary format
    (i.e., has an 8 byte header that should be ignored). Default: `false`.
* `numacopy::Bool` - Set to `true` to make NUMA local copies of the data. Default: `false`.
"""
function loadindex(
    dir,
    ::Type{T},
    dim,
    metric;
    datapath = joinpath(_prefix(dir), "data.bin"),
    allocator = direct_mmap,
    diskann_format = false,
    numacopy = false,
    # use_pq = false,
    # centroids_path = joinpath(dir, "centroids.bin"),
    # assignments_path = joinpath(dir, "assignments.bin"),
) where {T}
    if isa(dir, AbstractString)
        graph = GraphANN.load_bin(
            joinpath(dir, "graph.bin"),
            GraphANN.SuperFlatAdjacencyList{UInt32};
            writable = false,
        )
    elseif isa(dir, AbstractArray)
        graph = GraphANN.NumaAware(map(dir) do _dir
            GraphANN.load_bin(
                joinpath(_dir, "graph.bin"),
                GraphANN.SuperFlatAdjacencyList{UInt32};
                writable = false,
            )
        end)
    end

    data = loaddata(datapath, T, dim; allocator, diskann_format, numacopy)
    index = GraphANN.DiskANNIndex(graph, data, metric)
    return index
end

#####
##### Make Runner
#####

function make_runner(index::GraphANN.DiskANNIndex, search_window_size)
    return GraphANN.DiskANNRunner(
        index, search_window_size; executor = GraphANN.dynamic_thread
    )
end

#####
##### Search
#####

function convert_it(read_only_numpy_arr::PyCall.PyObject)
    # See https://github.com/JuliaPy/PyCall.jl/blob/master/src/pyarray.jl#L14-L24

    # instead of PyBUF_ND_STRIDED =  Cint(PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES)
    # See https://github.com/JuliaPy/PyCall.jl/blob/master/src/pybuffer.jl#L113
    pybuf = PyCall.PyBuffer(
        read_only_numpy_arr, PyCall.PyBUF_FORMAT | PyCall.PyBUF_ND | PyCall.PyBUF_STRIDES
    )

    T, native_byteorder = PyCall.array_format(pybuf)
    sz = size(pybuf)
    strd = PyCall.strides(pybuf)
    length(strd) == 0 && (sz = ())
    N = length(sz)
    isreadonly = pybuf.buf.readonly == 1
    info = PyCall.PyArray_Info{T,N}(
        native_byteorder, sz, strd, pybuf.buf.buf, isreadonly, pybuf
    )

    # See https://github.com/JuliaPy/PyCal.jl/blob/master/src/pyarray.jl#L123-L126
    return PyCall.PyArray{T,N}(read_only_numpy_arr, info)
end

function search(runner, index, queries::PyCall.PyObject, num_neighbors)
    pyarray = convert_it(queries)
    ptrvector = PtrVector{StaticArrays.SVector{size(pyarray, 2),eltype(pyarray)}}(
        pointer(pyarray), size(pyarray, 1)
    )
    return PyCall.PyReverseDims(search(runner, index, ptrvector, num_neighbors))
end

function search(runner, index, queries::PtrVector, num_neighbors)
    results = GraphANN.search(
        runner,
        index,
        queries;
        num_neighbors = num_neighbors,
        idmodifier = x -> (x - one(x)),
    )
    return results
end

# TODO: Generate search method to accept queries by pointer
