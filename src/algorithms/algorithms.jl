module Algorithms

export GraphParameters, generate_index
export GreedySearch, StartNode
export TreeSearcher

# stdlib
import LinearAlgebra
import Statistics
import Random

# local dependencies
using .._Base
using .._Graphs
using .._Trees
using .._Prefetcher
using .._Quantization

# deps
import DataStructures
import LightGraphs
import ProgressMeter
import Setfield
import StaticArrays: SVector, @SVector
import UnPack: @unpack, @pack!

import LightGraphs
graph_cost(meta::MetaGraph) = graph_cost(meta.graph, meta.data)
function graph_cost(graph, data::AbstractVector{T}) where {T}
    s = [Float64(evaluate(Euclidean(), data[edge.src], data[edge.dst])) for edge in LightGraphs.edges(graph)]
    return sum(s)
end

# exhaustive search
export exhaustive_search, exhaustive_search!
include("exhaustive.jl")

# diskann
include("diskann/search.jl")
include("diskann/build.jl")

# sptag
include("sptag/bktree.jl")
include("sptag/search.jl")
include("sptag/tptree.jl")
include("sptag/build.jl")

end # module
