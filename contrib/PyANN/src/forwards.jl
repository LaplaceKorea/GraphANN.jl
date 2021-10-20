#####
##### Load Index
#####

function loadgraph(path)
    return GraphANN.load_bin(
        path, GraphANN.SuperFlatAdjacencyList{UInt32}; writable = false
    )
end

function loaddata(path, ::Type{T}, dim::Integer) where {T}
    return GraphANN.load_bin(path, Vector{StaticArrays.SVector{dim,T}})
end

function loadindex(dir, ::Type{T}, dim, metric) where {T}
    graph = GraphANN.load_bin(
        joinpath(dir, "graph.bin"),
        GraphANN.SuperFlatAdjacencyList{UInt32};
        writable = false,
    )

    data = GraphANN.load_bin(joinpath(dir, "data.bin"), Vector{StaticArrays.SVector{dim,T}})

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
    results = GraphANN.search(runner, index, queries; num_neighbors = num_neighbors)
    results .-= one(eltype(results))
    return results
end

# TODO: Generate search method to accept queries by pointer
