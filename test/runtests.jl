using GraphANN
using Test

# Extra stdlib imports
using InteractiveUtils  # for `code_native`
using Statistics

# Extra helpful imports
using LightGraphs

import GraphANN: Neighbor

include("utils.jl")
include("spans.jl")
include("pm.jl")
include("threading.jl")
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
