module _Quantization

# local deps
using .._Base
using .._Graphs
using .._Points

# Special imports
import .._Points.deconstruct

# deps
import LightGraphs
import LightGraphs.SimpleGraphs.fadj
import ProgressMeter
import SIMD
import UnPack: @unpack


# Product Quantization
export PQTable, PQGraph
include("clustering.jl")
include("pq.jl")
include("pq_euclidean.jl")

end
