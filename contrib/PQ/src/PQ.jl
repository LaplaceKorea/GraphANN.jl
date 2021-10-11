module PQ

import GraphANN: GraphANN, MaybeThreadLocal

# stdlib
using LinearAlgebra
using Serialization
using Statistics

# deps
# import LoopVectorization
import SIMD
import StaticArrays: StaticVector, SVector, MVector, MMatrix
import LoopVectorization
using ProgressMeter: ProgressMeter
import UnPack: @unpack

include("atomic.jl")
include("distancetable.jl")
include("compress.jl")
include("fast.jl")

function __init__()
    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
end

end # module
