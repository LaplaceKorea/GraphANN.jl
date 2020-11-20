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

include("utils.jl")
include("spans.jl")
include("pm.jl")
include("bruteforce.jl")
include("points/euclidean.jl")
include("graphs/adjacency.jl")
include("graphs/graphs.jl")
include("test_minmax_heap.jl")
include("greedy.jl")

# Index building
include("index/index.jl")

# Loades
include("io/vecs.jl")

@testset "GraphANN.jl" begin
    # Write your tests here.
end
