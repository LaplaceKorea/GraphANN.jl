module GraphANN

import DataStructures
include("datastructures.jl")
include("minmax_heap.jl")

import Distances
import LightGraphs

include("graphs.jl")
include("algorithms.jl")

# Data representation
include("points/euclidean.jl")

end #module
