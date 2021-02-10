using GraphANN
using Test

# Extra stdlib imports
using InteractiveUtils  # for `code_native`
using Serialization
using Statistics

# Extra helpful imports
using LightGraphs
import SIMD
import StaticArrays: SVector, @SVector

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

include("base/pm.jl")
include("base/threading.jl")
include("base/minmax_heap.jl")
include("base/utils.jl")
include("base/bruteforce.jl")
# include("points/euclidean.jl")
#include("points/clustering.jl")
#include("quantization/clustering.jl")
#include("quantization/pq.jl")
# include("graphs/adjacency.jl")
# include("graphs/graphs.jl")
# include("query/greedy.jl")
include("algorithms/sptag.jl")

# # Query (comparison with DiskANN)
# include("query/query.jl")
#
# # Prefetch machinery
# include("prefetch/queue.jl")
# include("prefetch/prefetch.jl")
#
# # Index building
# include("index/index.jl")
#
# # Loads
# include("io/vecs.jl")
# include("io/sptag.jl")
#

