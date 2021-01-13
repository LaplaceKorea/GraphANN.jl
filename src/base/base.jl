# Basic functionality for the GraphANN module
# Exports for corresponding files are located right before their corresponding `include()`.
module _Base

# stdlib
import Mmap
import Statistics

# deps
import DataStructures
import ProgressMeter
import UnPack: @unpack

#####
##### Generic Distance
#####

export distance

"""
    distance(x, y)

Return the distance between points `x` and `y`.
"""
function distance end

"""
    distribute_distance(x)

Optional metric function. Distribute a necessary version of `x` to each thread.
"""
distribute_distance(x) = x

"""
    distance_prehook(metric, x)

Optional metric function. Perform necessary pre-distance computations on query `x`.
"""
distance_prehook(metric, x) = nothing

#####
##### Generic MetaGraph
#####

export MetaGraph

"""
    MetaGraph{G,D}

Grouping of a graph of type `G` and corresponding vertex data points of type `D`.
"""
struct MetaGraph{G,D}
    graph::G
    data::D
end

#####
##### Allocators
#####

export stdallocator, pmallocator
stdallocator(::Type{T}, dims...) where {T} = Array{T}(undef, dims...)
include("pm.jl")

#####
##### Threading
#####

export ThreadPool, ThreadLocal, TaskHandle
export getall, getpool, allthreads, dynamic_thread, on_threads
include("threading.jl")
distribute_distance(x::ThreadLocal) = x[]

#####
##### MinMaxHeap
#####

export BinaryMinMaxHeap, destructive_extract!, popmax!, popmin!, _unsafe_maximum, _unsafe_minimum
include("minmax_heap.jl")

#####
##### Spans
#####

export Span
include("spans.jl"); import ._Spans: Span

#####
##### Utilities
#####

export safe_maximum, donothing, printlnstyled, zero!, typemax!, cdiv, Map
export Neighbor, getid, getdistance, idequal
export RobinSet
export zeroas, medioid, nearest_neighbor
export recall
export prefetch, prefetch_llc, unsafe_prefetch
export BatchedRange, batched
include("utils.jl")

#####
##### Bruteforce Search
#####

export bruteforce_search
include("bruteforce.jl")

end
