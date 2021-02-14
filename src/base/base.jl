# Basic functionality for the GraphANN module
# Exports for corresponding files are located right before their corresponding `include()`.
module _Base

# stdlib
import Mmap

# deps
import DataStructures
import ProgressMeter
import SIMD
import StaticArrays: SVector
import TimerOutputs
import UnPack: @unpack

export @withtimer, gettimer, resettimer!
const timer = TimerOutputs.TimerOutput()
macro withtimer(description, expr)
    return :(TimerOutputs.@timeit timer $description $(esc(expr)))
end

gettimer() = timer
resettimer!() = TimerOutputs.reset_timer!(gettimer())

#####
##### Generic Distance
#####

export evaluate, search, searchall

"""
    evaluate(metric, x, y)

Return the distance between points `x` and `y`.
"""
function evaluate end

"""
    distribute(metric)

Optional metric function. Distribute a necessary version of `metric` to each thread.
Useful if metrics require local mutable scratch space.
"""
distribute(metric) = x

"""
    prehook(metric, x)

Optional metric function. Perform necessary pre-distance computations on query `x`.
"""
prehook(metric, x) = nothing

export Euclidean
include("euclidean.jl")

#####
##### Search Hooks
#####

function search end
function searchall end

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
export getall, getpool, allthreads, single_thread, dynamic_thread, on_threads
include("threading.jl")
#distribute_distance(x::ThreadLocal) = x[]

#####
##### MinMaxHeap
#####

export BinaryMinMaxHeap, destructive_extract!, popmax!, popmin!, _unsafe_maximum, _unsafe_minimum
include("minmax_heap.jl")

#####
##### Utilities
#####

export safe_maximum, donothing, printlnstyled, zero!, typemax!, cdiv, toeltype
export Neighbor, getid, getdistance, idtype, costtype
export RobinSet
export zeroas, medioid, nearest_neighbor
export recall
export prefetch, prefetch_llc, unsafe_prefetch
export BatchedRange, batched
export BoundedHeap, BoundedMinHeap, BoundedMaxHeap
include("utils.jl")

#####
##### Bruteforce Search
#####

export bruteforce_search, bruteforce_search!
include("bruteforce.jl")

#####
##### Partition utilities
#####

export PartitionUtil, partition!
include("partition.jl")

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

Neighbor(meta::MetaGraph, id, distance) = Neighbor{eltype(meta.graph)}(id, distance)

end
