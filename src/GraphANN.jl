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

# Includes
include("threading.jl"); using ._Threading
include("utils.jl")
include("spans.jl"); import .Spans: Span
include("pm.jl"); import .PM: pmmap
include("bruteforce.jl")
include("telemetry.jl"); import ._Telemetry: Telemetry, ifhasa

# Data representation
include("points/euclidean.jl")

include("graphs/graphs.jl")
include("query/greedy.jl")
include("index/index.jl")

# Prefetcher to increase performance
include("prefetch/prefetch.jl")

# Data loaders for various formats.
include("io/io.jl")

#####
##### Allocators
#####

# Allocator convenience functions
stdallocator(::Type{T}, dims...) where {T} = Array{T}(undef, dims...)

function pmallocator(::Type{T}, path::AbstractString, dims::Integer...) where {T}
    return pmmap(T, path, dims...)
end

# This is a partially applied version of the full allocator above.
# Use it like `f = pmallocator("path/to/dir")` to construct a function `f` that will
# have the same signature as the `stdallocator` above.
struct PMAllocator
    path::String
end
(f::PMAllocator)(type, dims...) = pmallocator(type, f.path, dims...)
pmallocator(path::AbstractString) = PMAllocator(path)

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
        window_size = 80,
        target_degree = 10,
        prune_threshold_degree = 15,
        prune_to_degree = 10,
    )

    return (;
        dataset,
        parameters,
    )
end

end # module
