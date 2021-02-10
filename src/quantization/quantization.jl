module _Quantization

# local deps
using .._Base
using .._Graphs
using .._Points

# Special imports
import .._Base: cast
import .._Points: squish, LazyWrap, LazyArrayWrap, Packed, set!

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

# Product Quantization
export PQTable, PQGraph
export encode
# #include("clustering.jl")
# include("clustering2.jl")
# include("pq.jl")
# include("pq_euclidean.jl")

end
