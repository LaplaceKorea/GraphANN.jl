module GraphANN

# We use StaticArrays pretty heavily in the code, so import the `SVector` symbol
# so users can access this type via `GraphANN.SVector`.
import StaticArrays: SVector

# Bootstrap
include("base/base.jl"); using ._Base
include("graphs/graphs.jl"); using ._Graphs
include("trees/trees.jl"); using ._Trees
include("points/points.jl"); using ._Points
include("prefetch/prefetch.jl"); using ._Prefetcher
include("quantization/quantization.jl"); using ._Quantization
include("io/io.jl"); using ._IO

# Core implementation
include("algorithms/algorithms.jl"); using .Algorithms

# function sweep(meta, start, queries, gt; num_neighbors = 5)
#     times, callbacks = Algorithms.latency_mt_callbacks()
#     for i in 10:10:200
#         algo = ThreadLocal(GreedySearch(i; costtype = costtype(meta.data)))
#         # warmup run.
#         GraphANN.searchall(algo, meta, start, queries; num_neighbors = num_neighbors, callbacks = callbacks)
#         # Actual run
#         rt = @elapsed ids = GraphANN.searchall(
#             algo,
#             meta,
#             start,
#             queries;
#             num_neighbors = num_neighbors,
#             callbacks = callbacks
#         )
#
#         _recall = mean(recall(gt, ids))
#         alltimes = reduce(vcat, getall(times))
#         empty!.(getall(times))
#
#         sorted_times = sort(alltimes)
#         latency_mean = mean(alltimes) / 1000
#         latency_999 = sorted_times[ceil(Int, 0.999 * length(sorted_times))] / 1000
#         latency_001 = sorted_times[floor(Int, 0.001 * length(sorted_times))] / 1000
#         qps = length(queries) / rt
#
#         print("$i\t\t")
#         print("$(round(qps; digits = 2))\t\t")
#         print("$(round(latency_mean; digits = 2))\t\t")
#         print("$(round(latency_999; digits = 2))\t\t")
#         print("$(round(latency_001; digits = 2))\t\t")
#         println(_recall)
#     end
# end

end # module
