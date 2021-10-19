module PyANN

#####
##### Exports
#####

# re-exports
export Euclidean, InnerProduct

# pyann implementstions
export loadindex

#####
##### Deps
#####

# local dependencies
import GraphANN
import PQ

# deps
import PyCall
import StaticArrays

#####
##### Forwards
#####

# constant mappings
const Euclidean = GraphANN.Euclidean
const InnerProduct = GraphANN.InnerProduct
include("forwards.jl")


# implementation
include("ptrarray.jl")


#####
##### API
#####

# Group together the index and algortihm runner into a single type on the Julia side.
struct IndexRunner{I,R}
    index::I
    runner::R
end

function batchquery(runner, index, queries::PyCall.PyObject, num_neighbors)
    pyarray = PyCall.PyArray(queries)
    ptrarray = PtrVector{StaticArrays.SVector{size(pyarray, 1), eltype(pyarray)}}(
        pointer(pyarray), size(pyarray, 2),
    )
    results = GC.@preserve pyarray GraphANN.search(runner, index, ptrarray, num_neighbors)
    return results
end

end # module
