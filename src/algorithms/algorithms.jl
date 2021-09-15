module Algorithms

export AbstractDiskANNRunner, DiskANNRunner, DiskANNIndex, DiskANNIndexParameters
export SPTAGRunner, SPTAGIndex

# stdlib
using LinearAlgebra: LinearAlgebra
using Statistics: Statistics
using Random: Random

# local dependencies
using .._Base
using .._Graphs
using .._Trees
using .._Prefetcher
using .._Clustering

# deps
using DataStructures: DataStructures
using LightGraphs: LightGraphs
using ProgressMeter: ProgressMeter
using Setfield: Setfield
import StaticArrays: SVector, @SVector
import UnPack: @unpack, @pack!

# utility for dispatching to the `ThreadLocal` constructor if desired.
threadlocal_wrap(::typeof(dynamic_thread), x) = ThreadLocal(x)
threadlocal_wrap(::typeof(single_thread), x) = x
threadlocal_wrap(tids::AbstractRange, x) = ThreadLocal(ThreadPool(tids), x)

# exhaustive search
export exhaustive_search
include("exhaustive.jl")

# diskann
include("diskann/maskwrap.jl")
include("diskann/search.jl")
include("diskann/build.jl")

# SPTAG
include("sptag/bktree.jl")
include("sptag/search.jl")
include("sptag/tptree.jl")
include("sptag/build.jl")

# Common Callback Implementations
include("callbacks.jl")

end # module
