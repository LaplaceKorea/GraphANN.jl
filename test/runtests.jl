using GraphANN
using Test

# Extra stdlib imports
using Random
using Serialization
using Statistics

# Extra helpful imports
using LightGraphs

import DataStructures
import SIMD
import StaticArrays: SVector, @SVector

import GraphANN: Neighbor

### Some tests fail when code-coverage is enabled (especially when related to timing)
# Introduce a method to skipping some checks if running with code coverage
const COVERAGE_ENABLED = !iszero(Base.JLOptions().code_coverage)
macro test_no_cc(expr)
    :(COVERAGE_ENABLED || @test($(esc(expr))))
end

### Paths to test datasets and reference indexes.
const datadir = joinpath(dirname(@__DIR__), "data")
const vecs_dir = joinpath(datadir, "vecs")
const diskann_dir = joinpath(datadir, "diskann")
const sptag_dir = joinpath(datadir, "sptag")

# SiftSmall in *vecs form.
const dataset_path = joinpath(vecs_dir, "siftsmall_base.fvecs")
const query_path = joinpath(vecs_dir, "siftsmall_query.fvecs")
const groundtruth_path = joinpath(vecs_dir, "siftsmall_groundtruth.ivecs")

# This is the path to an index constructed for siftsmall from the DiskANN open source code.
# The index was built with:
#
# R = 20
# L = 20
# α = 1.2
const diskann_index = joinpath(diskann_dir, "siftsmall_base_20_20_1.2.index")
const diskann_query_ids = joinpath(diskann_dir, "diskann_query_ids.jls")

# SPTAG generated files
const sptag_index = joinpath(sptag_dir, "siftsmall", "graph.bin")
const sptag_tree = joinpath(sptag_dir, "siftsmall", "tree.bin")

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
include("algorithms/sptag/bktree.jl")
include("algorithms/sptag/tptree.jl")
include("algorithms/sptag.jl")

# IO
include("io/vecs.jl")
include("io/sptag.jl")

