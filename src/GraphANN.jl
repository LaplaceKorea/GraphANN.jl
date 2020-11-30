module GraphANN

import DataStructures
include("minmax_heap.jl")

# stdlib
import Mmap

# Imports (avoid brining names into our namespace)
import LightGraphs
import ProgressMeter
import Setfield
import SIMD

# Explicit imports
import LightGraphs.SimpleGraphs: fadj
import UnPack: @unpack, @pack!

# Constants
const INDEX_BALANCE_FACTOR = 64
const ENABLE_THREADING = true

# Includes
include("utils.jl")
include("threading.jl")
include("spans.jl"); import .Spans: Span
include("pm.jl"); import .PM: pmmap
include("splitvector.jl"); import .SplitVectors: SplitVector
include("bruteforce.jl")

# Data representation
include("points/euclidean.jl")

include("graphs/graphs.jl")
include("query/query.jl")
include("index/index.jl")

# Prefetcher to increase performance
include("prefetch/prefetch.jl")

# Data loaders for various formats.
include("io/io.jl")


# Allocator convenience functions
stdallocator(::Type{T}, dims...) where {T} = Array{T}(undef, dims...)

function pmallocator(::Type{T}, path::AbstractString, dims::Integer...) where {T}
    return pmmap(T, path, dims...)
end

# This is a partially applied version of the full allocator above.
# Use it like `f = pmallocator("path/to/dir")` to construct a function `f` that will
# have the same signature as the `stdallocator` above.
pmallocator(path::AbstractString) = (type, dims...) -> pmallocator(type, path, dims...)

# function splitallocator(::Type{T}, path::AbstractString, dramsize, dim) where {T}
#     requested_size = sizeof(T) * dim
#     dram_alloc_len = div(min(dramsize, requested_size), sizeof(T))
#     pm_alloc_len = max(0, div(requested_size - dramsize, sizeof(T)))
#
#     return SplitVector{T}(
#         undef,
#         dram_alloc_len,
#         stdallocator,
#         dram_alloc_len,
#         pmallocator(path)
#     )
# end
#
# function splitallocator(path::AbstractString, dramsize)
#     return (type, dim) -> splitallocator(type, path, dramsize, dim)
# end

#####
##### Misc development functions
#####

siftsmall() = joinpath(dirname(@__DIR__), "data", "siftsmall_base.fvecs")
function _prepare(path = siftsmall(); allocator = stdallocator, maxlines = nothing)
    dataset = load_vecs(
        Euclidean{128,UInt8},
        path;
        maxlines = maxlines,
        allocator = allocator
    )

    parameters = GraphParameters(
        alpha = 1.2,
        window_size = 200,
        target_degree = 128,
        prune_threshold_degree = 140,
        prune_to_degree = 120,
    )

    return (;
        dataset,
        parameters,
    )
end

function sweep(
    meta::MetaGraph,
    start_node,
    queries::AbstractVector;
    num_neighbors = 1,
    buf_range = 10:5:120,
)
    all_times = []
    all_ids = []

    for window_size in buf_range
        @show window_size
        algo = GreedySearch(window_size)

        # One warm up run to allocate data structures.
        searchall(algo, meta, start_node, queries; num_neighbors = num_neighbors)

        ids, times = searchall(
            algo,
            meta,
            start_node,
            queries;
            num_neighbors = num_neighbors
        )
        push!(all_times, times)
        push!(all_ids, ids)
    end
    return all_ids, all_times
end

end #module
