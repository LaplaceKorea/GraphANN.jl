module _Points

# local deps
using .._Base
using .._Graphs
import .._IO

# stdlib
import Random

# deps
import LightGraphs
import LightGraphs.SimpleGraphs: fadj
import ProgressMeter
import SIMD
using StaticArrays
import UnPack: @unpack

export Euclidean
include("euclidean.jl")

end # module
