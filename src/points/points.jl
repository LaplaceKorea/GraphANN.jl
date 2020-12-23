module _Points


# local deps
using .._Base
using .._Graphs
import .._IO

# deps
import HDF5
import LightGraphs
import LightGraphs.SimpleGraphs: fadj
import ProgressMeter
import SIMD
using StaticArrays
import UnPack: @unpack

export SIMDLanes
include("simd.jl")

export Euclidean
include("euclidean.jl")

end # module
