module GraphANN

import DataStructures
include("minmax_heap.jl")

# Imports (avoid brining names into our namespace)
import Distances
import LightGraphs
import ProgressMeter
import Setfield
import SIMD

# Explicit imports
import LightGraphs.SimpleGraphs: fadj
import UnPack: @unpack, @pack!

# Includes
include("utils.jl")
include("threading.jl")

# Data representation
include("points/euclidean.jl")

include("graphs/graphs.jl")
include("algorithms.jl")
include("index/index.jl")

# Data loaders for various formats.
include("io/io.jl")

#####
##### Misc development functions
#####

siftsmall() = joinpath(dirname(@__DIR__), "data", "siftsmall_base.fvecs")
function _prepare(path = siftsmall())
    dataset = load_vecs(Euclidean{128,UInt8}, path)
    #dataset = load_vecs(Euclidean{128,Float32}, path)

    parameters = GraphParameters(
        alpha = 1.2,
        window_size = 50,
        target_degree = 128,
        prune_threshold_degree = 150,
        prune_to_degree = 100,
    )

    # parameters = GraphParameters(
    #     alpha = 1.2,
    #     window_size = 50,
    #     target_degree = 50,
    #     prune_threshold_degree = 70,
    #     prune_to_degree = 30,
    # )

    return (;
        dataset,
        parameters,
    )
end

end #module
