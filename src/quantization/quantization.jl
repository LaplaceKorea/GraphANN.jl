module _Quantization

# local deps
using .._Base
using .._Graphs
using .._Points

# Special imports
import .._Points: cast, squish, LazyWrap, LazyArrayWrap, Packed, set!

# deps
import LightGraphs
import LightGraphs.SimpleGraphs.fadj
import ProgressMeter
import SIMD
import UnPack: @unpack


# Product Quantization
export PQTable, PQGraph
export encode
include("clustering.jl")
include("pq.jl")
include("pq_euclidean.jl")

end
