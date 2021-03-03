module GraphANN

# We use StaticArrays pretty heavily in the code, so import the `SVector` symbol
# so users can access this type via `GraphANN.SVector`.
import StaticArrays: SVector

# Bootstrap
include("base/base.jl"); using ._Base
include("graphs/graphs.jl"); using ._Graphs
include("trees/trees.jl"); using ._Trees
include("prefetch/prefetch.jl"); using ._Prefetcher
include("clustering/clustering.jl"); using ._Clustering
include("io/io.jl"); using ._IO

# Core implementation
include("algorithms/algorithms.jl"); using .Algorithms

# For examples
function sample_dataset()
    path = joinpath(dirname(@__DIR__), "data", "vecs", "siftsmall_base.fvecs")
    return load_vecs(SVector{128,Float32}, path)
end

############################################################################################
# function __test(data)
#     numsamples = 10000
#     leafsize = 500
#     numtrials = 200
#     num_neighbors = 100
#     permutation = UInt32.(eachindex(data))
#     D = costtype(data)
#
#     # Just group together every 500 data points.
#     base_distances = Matrix{D}(undef, num_neighbors, length(data))
#     for range in GraphANN.batched(1:length(data), leafsize)
#         erunner = GraphANN.Algorithms.ExhaustiveRunner(
#             GraphANN.Neighbor{UInt32,D},
#             length(range),
#             num_neighbors;
#             costtype = D,
#         )
#
#         dataview = view(data, range)
#         gt = GraphANN.Algorithms.search!(erunner, dataview, dataview; meter = nothing)
#         base_view = view(base_distances, :, range)
#         base_view .= GraphANN.getdistance.(gt)
#     end
#
#     ranges = GraphANN.Algorithms.partition!(
#         data,
#         permutation,
#         Val(4);
#         leafsize = leafsize,
#         numtrials = numtrials,
#         numsamples = numsamples,
#         init = true,
#         single_thread_threshold = div(length(data), 10),
#     )
#
#     clustered_distances = Matrix{D}(undef, num_neighbors, length(data))
#     for range in ranges
#         # Pre-allocate the result matrix for `bruteforce_search` with eltype `Neighbor`.
#         # This will ensure that we get the distances.
#         erunner = GraphANN.Algorithms.ExhaustiveRunner(
#             GraphANN.Neighbor{UInt32,D},
#             length(range),
#             num_neighbors;
#             costtype = D,
#         )
#         dataview = GraphANN.Algorithms.doubleview(data, permutation, range)
#         gt = GraphANN.Algorithms.search!(erunner, dataview, dataview; meter = nothing)
#
#         # Note: We need to translate from the indices returned by `bruteforce_search!` to
#         # original indices in the dataset.
#         # The position-wise corresponding global indices can be found by the `viewperm`
#         # function.
#         permview = view(permutation, range)
#         clustered_view = view(clustered_distances, :, permview)
#         clustered_view .= GraphANN.getdistance.(gt)
#     end
#     return base_distances, clustered_distances
# end

end # module
