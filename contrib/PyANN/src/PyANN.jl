module PyANN

#####
##### Exports
#####

# re-exports
export Euclidean, InnerProduct

# pyann implementstions
export loadindex, search

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

# implementation
include("ptrarray.jl")


# constant mappings
const Euclidean = GraphANN.Euclidean
const InnerProduct = GraphANN.InnerProduct
include("forwards.jl")

end # module
