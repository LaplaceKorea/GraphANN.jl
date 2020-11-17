module GraphANN

import DataStructures
include("minmax_heap.jl")

# Imports (avoid brining names into our namespace)
import Distances
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

# Data representation
include("points/euclidean.jl")

include("graphs/graphs.jl")
include("algorithms.jl")
include("index/index.jl")

# Data loaders for various formats.
include("io/io.jl")

#####
##### Misc development functions
#####

siftsmall() = joinpath(dirname(@__DIR__), "data", "siftsmall_base.fvecs")
function _prepare(path = siftsmall())
    dataset = load_vecs(Euclidean{128,UInt8}, path)

    parameters = GraphParameters(
        alpha = 1.2,
        window_size = 50,
        target_degree = 128,
        prune_threshold_degree = 150,
        prune_to_degree = 100,
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
    buf_range = 5:5:120,
)
    all_times = []
    all_ids = []

    for window_size in buf_range
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
