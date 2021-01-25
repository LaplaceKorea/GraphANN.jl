module GraphANN

# Bootstrap
include("base/base.jl"); using ._Base
include("graphs/graphs.jl"); using ._Graphs
include("trees/trees.jl"); using ._Trees
include("io/io.jl"); using ._IO
include("points/points.jl"); using ._Points
include("prefetch/prefetch.jl"); using ._Prefetcher
include("quantization/quantization.jl"); using ._Quantization

# Core implementation
include("algorithms/algorithms.jl")
using .Algorithms

#####
##### Misc development functions
#####

# siftsmall() = joinpath(dirname(@__DIR__), "data", "siftsmall_base.fvecs")
# function _prepare(path = siftsmall(); allocator = stdallocator, maxlines = nothing)
#     dataset = load_vecs(
#         Euclidean{128,UInt8},
#         path;
#         maxlines = maxlines,
#         allocator = allocator
#     )
#
#     parameters = GraphParameters(
#         alpha = 1.2,
#         window_size = 80,
#         target_degree = 10,
#         prune_threshold_degree = 15,
#         prune_to_degree = 10,
#     )
#
#     return (;
#         dataset,
#         parameters,
#     )
# end

end # module
