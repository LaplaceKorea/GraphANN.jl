module GraphANN

import DataStructures
include("datastructures.jl")
include("minmax_heap.jl")

import Distances
import LightGraphs

include("graphs.jl")
include("algorithms.jl")

# Data representation
include("points/euclidean.jl")

# Serialization and Deserialization
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

end #module
