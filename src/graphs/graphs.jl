"""
    MetaGraph{G,D}

Grouping of a graph of type `G` and corresponding vertex data points of type `D`.
"""
struct MetaGraph{G,D}
    graph::G
    data::D
end

include("adjacency.jl")
include("unidirected.jl")
include("io.jl")
include("generators.jl")

# random utility functions
function edgecheck(A::UniDirectedGraph, B::UniDirectedGraph)
    @assert LightGraphs.nv(A) == LightGraphs.nv(B)
    @assert LightGraphs.ne(A) == LightGraphs.ne(B)

    pmeter = ProgressMeter.Progress(LightGraphs.ne(A), 1)

    for (a, b) in zip(LightGraphs.edges(A), LightGraphs.edges(B))
        a != b && error("Uh Oh")
        ProgressMeter.next!(pmeter)
    end
    ProgressMeter.finish!(pmeter)
end

