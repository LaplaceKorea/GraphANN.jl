module GraphANN

import DataStructures
include("minmax_heap.jl")

# Imports (avoid brining names into our namespace)
import Distances
import LightGraphs
import ProgressMeter
import SIMD

# Explicit imports
import LightGraphs.SimpleGraphs: fadj

# Import names
using UnPack

include("utils.jl")

# Data representation
include("points/euclidean.jl")

include("graphs.jl")
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

function to_euclidean(x::AbstractMatrix{T}) where {T}
    dim = size(x,1)
    x = reshape(x, :)
    x = reinterpret(GraphANN.Euclidean{dim,T}, x)
    return collect(x)
end

function _prepare()
    dataset = load_vecs(joinpath(@__DIR__, "..", "data", "siftsmall_base.fvecs"))
    dataset = to_euclidean(dataset)

    parameters = GraphParameters(1.2, 70, 50)
    return (;
        dataset,
        parameters,
    )
end

end #module
