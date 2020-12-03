"""
    MetaGraph{G,D}

Grouping of a graph of type `G` and corresponding vertex data points of type `D`.
"""
struct MetaGraph{G,D}
    graph::G
    data::D
end

# Top level file - include the implementation files.
include("adjacency.jl")
include("unidirected.jl")
include("io.jl")
include("generators.jl")

