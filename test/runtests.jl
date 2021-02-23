using GraphANN
using Test

# Extra stdlib imports
using Random
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

# Base
include("base/pm.jl")
include("base/utils.jl")
include("base/threading.jl")
include("base/minmax_heap.jl")
include("base/euclidean.jl")
include("base/partition.jl")

# Graphs
include("graphs/adjacency.jl")
include("graphs/graphs.jl")

# Trees
include("trees/trees.jl")

# Clustering
include("clustering/kmeans.jl")

# Prefetch
include("prefetch/queue.jl")
include("prefetch/prefetch.jl")

# Algorithms
include("algorithms/exhaustive.jl")
include("algorithms/diskann.jl")
include("algorithms/sptag.jl")

# IO
include("io/vecs.jl")
include("io/sptag.jl")

