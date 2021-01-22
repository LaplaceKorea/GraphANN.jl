module _Points

# local deps
using .._Base
import .._IO

# stdlib
import Random

# deps
import SIMD
import StaticArrays: SVector, @SVector
import UnPack: @unpack

export Euclidean
include("euclidean.jl")
include("experimental.jl")

end # module
