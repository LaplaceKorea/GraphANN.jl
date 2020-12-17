module _Points

export Euclidean

# local deps
using .._Base
import .._IO

# deps
import ProgressMeter
import SIMD
import UnPack: @unpack

include("euclidean.jl")
include("clustering.jl")

end
