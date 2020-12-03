using GraphANN
using Test

# Extra stdlib imports
using InteractiveUtils  # for `code_native`
using Statistics

# Extra helpful imports
using LightGraphs

import GraphANN: Neighbor

const datadir = joinpath(dirname(@__DIR__), "data")
const dataset_path = joinpath(datadir, "siftsmall_base.fvecs")
const query_path = joinpath(datadir, "siftsmall_query.fvecs")
const groundtruth_path = joinpath(datadir, "siftsmall_groundtruth.ivecs")

# This is the path to an index constructed for siftsmall from the DiskANN open source code.
# The index was built with:
#
# R = 70
# L = 75
# α = 1.2
#
# When running inference with DiskANN finding the 5 nearest neighbors with a window size
# of 10, the expected accuracy is ** 98.6% **
const diskann_index = joinpath(datadir, "siftsmall_base_70_75_1.2.index")

include("utils.jl")
include("spans.jl")
include("pm.jl")
include("bruteforce.jl")
include("points/euclidean.jl")
include("graphs/adjacency.jl")
include("graphs/graphs.jl")
include("test_minmax_heap.jl")
include("query/telemetry.jl")
include("query/greedy.jl")

# Query (comparison with DiskANN)
include("query/query.jl")

# Prefetch machinery
include("prefetch/queue.jl")

# Index building
include("index/index.jl")

# Loades
include("io/vecs.jl")

