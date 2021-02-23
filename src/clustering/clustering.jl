module _Clustering

# local deps
using .._Base

# deps
import StaticArrays: SVector
import UnPack: @unpack

export KMeansRunner, kmeans
include("kmeans.jl")

end
