module PQ

import GraphANN: GraphANN, MaybeThreadLocal

# stdlib
using LinearAlgebra
using Statistics

# deps
import LoopVectorization
import SIMD
import StaticArrays: StaticVector, SVector, MVector, MMatrix
using ProgressMeter: ProgressMeter
import UnPack: @unpack

include("distancetable.jl")
include("compress.jl")
include("fast.jl")

end # module
