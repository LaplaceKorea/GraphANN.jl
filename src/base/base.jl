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

export evaluate, prehook, build, search, search!

"""
    evaluate(metric, x, y)

Return the distance with respect to `metric` between points `x` and `y`.
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

function build end
function search end
function search! end

#####
##### Allocators
#####

export stdallocator, pmallocator

"""
    stdallocator

Singleton type that is the default implementation for allocators.
When invoked, simply constructs an uninitialized standard Julia array.

# Example
```jldoctest
julia> allocator = GraphANN.stdallocator;

julia> A = allocator(Int64, 2, 2);

julia> typeof(A)
Matrix{Int64} (alias for Array{Int64, 2})

julia> size(A)
(2, 2)
```
"""
stdallocator(::Type{T}, dims...) where {T} = Array{T}(undef, dims...)
include("pm.jl")

#####
##### Threading
#####

export ThreadPool, ThreadLocal, MaybeThreadLocal, TaskHandle
export getall, getlocal, getpool, allthreads, single_thread, dynamic_thread, on_threads
export threadcopy
include("threading.jl")

#####
##### MinMaxHeap
#####

export BinaryMinMaxHeap, destructive_extract!, popmax!, popmin!, _unsafe_maximum, _unsafe_minimum
include("minmax_heap.jl")

#####
##### Utilities
#####

export safe_maximum, donothing, printlnstyled, always_false, zero!, typemax!, cdiv, toeltype, clog2
export Neighbor, getid, getdistance, idtype, costtype
export RobinSet
export zeroas, medioid, nearest_neighbor
export recall
export prefetch, prefetch_llc, unsafe_prefetch
export BatchedRange, batched
export Keeper, KeepLargest, KeepSmallest
include("utils.jl")

#####
##### Partition utilities
#####

export PartitionUtil, partition!
include("partition.jl")

# TODO: Remove
export MetaGraph
struct MetaGraph{G,D}
    graph::G
    data::D
end

end
