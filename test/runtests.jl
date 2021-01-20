using GraphANN
using Test

# Extra stdlib imports
using InteractiveUtils  # for `code_native`
using Serialization
using Statistics

# Extra helpful imports
using LightGraphs
import SIMD

import GraphANN: Neighbor

const datadir = joinpath(dirname(@__DIR__), "data")
const dataset_path = joinpath(datadir, "siftsmall_base.fvecs")
const query_path = joinpath(datadir, "siftsmall_query.fvecs")
const groundtruth_path = joinpath(datadir, "siftsmall_groundtruth.ivecs")

# This is the path to an index constructed for siftsmall from the DiskANN open source code.
# The index was built with:
#
# R = 20
# L = 20
# α = 1.2
const diskann_index = joinpath(datadir, "siftsmall_base_20_20_1.2.index")
const diskann_query_ids = joinpath(datadir, "diskann_query_ids.jls")

# include("threading.jl")
# include("utils.jl")
# include("spans.jl")
# include("pm.jl")
# include("bruteforce.jl")
# include("points/euclidean.jl")
# #include("points/clustering.jl")
# include("quantization/clustering.jl")
# include("quantization/pq.jl")
# include("graphs/adjacency.jl")
# include("graphs/graphs.jl")
# include("test_minmax_heap.jl")
# include("query/greedy.jl")

# # Query (comparison with DiskANN)
# include("query/query.jl")
#
# # Prefetch machinery
# include("prefetch/queue.jl")
# include("prefetch/prefetch.jl")

# Index building
include("index/index.jl")

# Loads
include("io/vecs.jl")

