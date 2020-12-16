module _Graphs

# local deps
using .._Base

# deps
import LightGraphs
import ProgressMeter
import UnPack: @unpack

# explicit imports
import LightGraphs.SimpleGraphs.fadj

# Top level file - include the implementation files.
export DefaultAdjacencyList, FlatAdjacencyList, DenseAdjacencyList
include("adjacency.jl")

export UniDirectedGraph
include("unidirected.jl")

export save, load
include("io.jl")

export random_regular
include("generators.jl")

end #module
