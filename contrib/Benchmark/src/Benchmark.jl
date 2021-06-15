module Benchmark

using GraphANN: GraphANN

# stdlib
using Serialization
using Statistics

# deps
using DataFrames
using DataStructures
using ProgressMeter

using PrettyTables: PrettyTables
import UnPack: @unpack, @pack!
import HugepageMmap

# paths
const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const SCRATCH = joinpath(PKGDIR, "data")

# init
makescratch() = ispath(SCRATCH) || mkpath(SCRATCH)
function __init__()
    return makescratch()
end
include("types.jl")

function go()
    println("Using $(Threads.nthreads()) threads")
    record = Record("new_run_data.jls")
    # Load data
    println("Loading Data")
    _data = GraphANN.load_bin("/mnt/pm1/public/sift1b.bin", Vector{GraphANN.SVector{128,UInt8}})
    data = HugepageMmap.hugepage_mmap(GraphANN.SVector{128,UInt8}, length(_data), HugepageMmap.PageSize1G())
    GraphANN.dynamic_thread(eachindex(data), 64) do i
        data[i] = _data[i]
    end

    # Load graph
    println("Loading Graph")
    graph = GraphANN.load_bin("/mnt/pm1/public/sift1b_index/", GraphANN.UniDirectedGraph{UInt32, GraphANN.DenseAdjacencyList{UInt32}})

    # Create index
    index = GraphANN.DiskANNIndex(graph, data)

    # Load queries and groundtruth
    queries = GraphANN.load_vecs(GraphANN.SVector{128,UInt8}, "/data/sift/queries/queries_1m.bvecs")
    groundtruth = GraphANN.load_vecs("/data/sift/groundtruth/queries_1m/sift1b.ivecs"; groundtruth = true)

    println("Running Benchmark")
    run_diskann(record, index, queries, groundtruth; multithread = true)
end

# routines
include("sptag.jl")
include("diskann.jl")

end # module
