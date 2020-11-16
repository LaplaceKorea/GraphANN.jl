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

# Import names
using UnPack

# Allow us to turn off threading so we can inspect routines a bit better.
const ENABLE_THREADING = true

@static if ENABLE_THREADING
    macro threads(expr)
        return :(Threads.@threads $(expr))
    end
else
    macro threads(expr)
        return :($(esc(expr)))
    end
end


include("utils.jl")

# Data representation
include("points/euclidean.jl")

include("graphs/graphs.jl")
include("algorithms.jl")
include("index/index.jl")

# Data loaders for various formats.
include("io/io.jl")

function prepare()
    data_path = "/home/stg/projects/diskann-experiments/experiments/DiskANN/data/processed/sift1m/sift_base"
    index_path = "/home/stg/projects/diskann-experiments/experiments/DiskANN/data/index/sift1m/sift_base_70_75_1.2.index"
    query_path = "/home/stg/projects/diskann-experiments/experiments/DiskANN/data/processed/sift1m/sift_query"
    #groundtruth_path = "/home/stg/projects/diskann-experiments/experiments/DiskANN/data/processed/sift1m/sift_groundtruth"

    loader = DiskANNLoader()
    data = load_data(Float32, loader, data_path)
    queries = load_data(Float32, loader, query_path)
    graph = load_graph(loader, index_path, length(data))

    meta = MetaGraph(graph, data)
    algo = GreedySearch(100)
    start = entry_point(data)

    return (;
        algo,
        meta,
        start,
        queries,
    )
end

siftsmall() = joinpath(dirname(@__DIR__), "data", "siftsmall_base.fvecs")
function _prepare(path = siftsmall())
    dataset = load_vecs(Euclidean{128,UInt8}, path)
    #dataset = load_vecs(Euclidean{128,Float32}, path)

    #parameters = GraphParameters(1.2, 30, 20, 0.75)
    #parameters = GraphParameters(1.2, 128, 50, 0.75)
    parameters = GraphParameters(1.2, 128, 50, 0.8, 1.2)
    return (;
        dataset,
        parameters,
    )
end

end #module
