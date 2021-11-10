# Basic functionality for the GraphANN module
# Exports for corresponding files are located right before their corresponding `include()`.
module _Base

# stdlib
import Libdl
import Mmap

# deps
import DataStructures
import HugepageMmap
import MacroTools
import ProgressMeter
import SIMD
import StaticArrays: SVector
import TimerOutputs
import UnPack: @unpack

#####
##### Initialization
#####

function __init__()
    if get(ENV, "JULIA_EXCLUSIVE", 0) != 0
        @info "Running in an exclusive environment. Populating thread affinities."

        # Assign thread ids to NUMA nodes
        resize!(NUMAMAP, Threads.nthreads())
        on_threads(allthreads()) do
            tid = Threads.threadid()
            node = @ccall libnuma.numa_node_of_cpu((tid - 1)::Cint)::Cint
            NUMAMAP[tid] = node + 1
        end

        # Find the number of NUMA nodes
        NUM_NUMA_NODES[] = length(unique(NUMAMAP))
    end
end

export @withtimer, gettimer, resettimer!
const timer = TimerOutputs.TimerOutput()
macro withtimer(description, expr)
    return :(TimerOutputs.@timeit timer $description $(esc(expr)))
end

gettimer() = timer
resettimer!() = TimerOutputs.reset_timer!(gettimer())

#####
##### Compiler Hints
#####

export @_nointerleave_meta
macro _interleave_meta(n)
    return Expr(
        :loopinfo,
        (Symbol("llvm.loop.interleave.count"), n),
        (Symbol("llvm.loop.unroll.disable"), 2),
    )
end

#####
##### Generic Distance
#####

export MaybePtr, evaluate, prehook, build, search, search!, ordering
const MaybePtr{T} = Union{T,Ptr{<:T}}
ordering(metric) = Base.Forward

"""
    evaluate(metric, x::MaybePtr, y::MaybePtr)

Return the distance with respect to `metric` between points `x` and `y`.
"""
function evaluate end

"""
    prehook(metric, x::MaybePtr)

Optional metric function. Perform necessary pre-distance computations on query `x`.
"""
prehook(metric, x) = nothing

export Euclidean, InnerProduct
include("metric.jl")

#####
##### Search Hooks
#####

function build end
function search end
function search! end

#####
##### Allocators
#####

export stdallocator, pmallocator, hugepage_1gib_allocator, hugepage_2mib_allocator

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

"""
    hugepage_1gib_allocator

**EXPERIMENTAL**: Allocator implementation that backs allocated vectors using 1 GiB
hugepages. Note that the hugepages must be allocated beforehand using a tool like
`hugeadm`. See: `https://github.com/hildebrandmw/HugepageMmap.jl`.

See also: [`hugepage_2mib_allocator`](@ref)
"""
function hugepage_1gib_allocator(::Type{T}, dims::Integer...) where {T}
    return HugepageMmap.hugepage_mmap(T, HugepageMmap.PageSize1G(), dims...)
end

"""
    hugepage_2mib_allocator

**EXPERIMENTAL**: Allocator implementation that backs allocated vectors using 2 MiB
hugepages. Note that the hugepages must be allocated beforehand using a tool like
`hugeadm`. See: `https://github.com/hildebrandmw/HugepageMmap.jl`.

See also: [`hugepage_1gib_allocator`](@ref)
"""
function hugepage_2mib_allocator(::Type{T}, dims::Integer...) where {T}
    return HugepageMmap.hugepage_mmap(T, HugepageMmap.PageSize2M(), dims...)
end

#####
##### Threading
#####

export ThreadPool, ThreadLocal, MaybeThreadLocal, TaskHandle
export getall, getlocal, getpool, allthreads, single_thread, dynamic_thread, on_threads
export threadcopy
include("threading.jl")

#####
##### Utilities
#####

export safe_maximum,
    donothing, printlnstyled, always_false, zero!, typemax!, cdiv, toeltype, clog2
export Neighbor, getid, getdistance, idtype, costtype
export RobinSet, ifmissing!
export zeroas, medioid, nearest_neighbor
export recall
export prefetch, prefetch_llc, unsafe_prefetch
export BatchedRange, batched
export Keeper, KeepLargest, KeepSmallest
export destructive_extract!
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

#####
##### FastSet
#####

export FastSet
include("fastset.jl")

#####
##### numa
#####

export NumaAware, @numalocal, MaybeNumaAware
include("numa.jl")

end
