module _Quantization

# local deps
using .._Base
using .._Graphs

# Special imports
import .._Base: cast

# deps
import LightGraphs
import LightGraphs.SimpleGraphs.fadj
import ProgressMeter
import SIMD
import StaticArrays: SVector
import UnPack: @unpack

# Widen data types to 64 bits
widen64(::Type{<:Unsigned}) = UInt64
widen64(::Type{<:Signed}) = Int64
widen64(::Type{<:AbstractFloat}) = Float64
widen64(x::T) where {T} = convert(widen64(T), x)

include("clustering.jl")

end
