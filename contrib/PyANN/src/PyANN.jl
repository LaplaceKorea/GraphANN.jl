module PyANN

#####
##### Exports
#####

# re-exports
export Euclidean, InnerProduct
export stdallocator, pmallocator, hugepage_1gib_allocator, hugepage_2mib_allocator

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
include("forwards.jl")

# constant mappings
const Euclidean = GraphANN.Euclidean
const InnerProduct = GraphANN.InnerProduct
const stdallocator = GraphANN.stdallocator
const pmallocator = GraphANN.pmallocator
const hugepage_1gib_allocator = GraphANN.hugepage_1gib_allocator
const hugepage_2mib_allocator = GraphANN.hugepage_2mib_allocator

end # module
