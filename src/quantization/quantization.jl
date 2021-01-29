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

# Widen data types, but no bigger than 64-bits.
const NoWidenTypes = Union{Float64,Int64,UInt64}
maybe_widen(x) = widen(x)
maybe_widen(x::T) where {T <: NoWidenTypes} = x
maybe_widen(::Type{T}) where {T} = widen(T)
maybe_widen(::Type{T}) where {T <: NoWidenTypes} = T

# Product Quantization
export PQTable, PQGraph
export encode
include("clustering.jl")
include("pq.jl")
include("pq_euclidean.jl")

end
