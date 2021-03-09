module GraphANN

import Statistics

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DATADIR = joinpath(PKGDIR, "data")

# We use StaticArrays pretty heavily in the code, so import the `SVector` symbol
# so users can access this type via `GraphANN.SVector`.
import StaticArrays: SVector

# Bootstrap
include("base/base.jl"); using ._Base
include("graphs/graphs.jl"); using ._Graphs
include("trees/trees.jl"); using ._Trees
include("prefetch/prefetch.jl"); using ._Prefetcher
include("clustering/clustering.jl"); using ._Clustering
include("dataset.jl")

# Core implementation
include("algorithms/algorithms.jl"); using .Algorithms
include("io/io.jl"); using ._IO

# For examples
const VECSDIR = joinpath(DATADIR, "vecs")
sample_dataset() = load_vecs(SVector{128,Float32}, joinpath(VECSDIR, "siftsmall_base.fvecs"))
sample_queries() = load_vecs(SVector{128,Float32}, joinpath(VECSDIR, "siftsmall_query.fvecs"))
sample_groundtruth() = load_vecs(joinpath(VECSDIR, "siftsmall_groundtruth.ivecs"); groundtruth = true)

############################################################################################
function __run(index, queries, groundtruth, windows; num_neighbors = 5)
    times, callbacks = Algorithms.latency_mt_callbacks()

    results = Any[]
    for window in windows
        runner = ThreadLocal(DiskANNRunner(index, window))
        # Warmup
        @time ids = searchall(runner, index, queries; num_neighbors, callbacks)
        empty!.(getall(times))
        rt = @elapsed ids = searchall(runner, index, queries; num_neighbors, callbacks)
        recall = Statistics.mean(_Base.recall(groundtruth, ids))
        alltimes = reduce(vcat, getall(times))

        latency_mean = Statistics.mean(alltimes) / 1000
        latency_999 = sort(alltimes)[ceil(Int, 0.999 * length(alltimes))] / 1000
        nt = (
            recall = recall,
            qps = length(queries) / rt,
            latency_mean = latency_mean,
            latency_999 = latency_999,
        )
        push!(results, nt)
    end
    return results
end

end # module
