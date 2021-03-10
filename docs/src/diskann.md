# DiskANN

The example below will walk through a complete example using the example dataset found in
the `data` folder of this repository.

## Index Building

First, we need to load a dataset.
We could use `GraphANN.sample_dataset()` to load the sample dataset for us, but we'll manually load the dataset to demonstrate how this works.
The sample dataset is the `siftsmall` dataset stored in the [`fvecs`](http://corpus-texmex.irisa.fr/) form.
```jldoctest diskann-example
julia> using GraphANN

julia> path = joinpath(GraphANN.DATADIR, "vecs", "siftsmall_base.fvecs");

julia> dataset = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, path);
```
With the dataset loaded we next select parameters for graph construction.
Reference [`DiskANNIndexParameters`](@ref GraphANN.DiskANNIndexParameters) for a full list of index building options and descriptions.
```jldoctest diskann-example
julia> parameters = GraphANN.DiskANNIndexParameters(alpha = 1.2, window_size = 128, target_degree = 64, prune_threshold_degree = 80, prune_to_degree = 64)
GraphANN.Algorithms.DiskANNIndexParameters(1.2, 128, 64, 80, 64)
```
Finally, we build a [`DiskANNIndex`](@ref GraphANN.DiskANNIndex) index using [`build`](@ref GraphANN.build(::Any, GraphANN.DiskANNIndexParameters)).
```jldoctest diskann-example; filter = r"\w[0-9]+\}"
# Set the RNG seed for reproducibility.
# Your numbers may vary slightly depending on the number of threads being used.
julia> import Random; Random.seed!(1234);

julia> index = GraphANN.build(dataset, parameters; no_progress = true)
DiskANNIndex(10000 data points. Entry point index: 3733)

julia> index.graph
{10000, 409562} directed simple UInt32 graph

julia> using LightGraphs; maximum(outdegree(index.graph))
64
```

## Index Query

With the graph generated, it's time to start running queries.
First, we need to pre-allocate some scratch space in the form of a [`DiskANNRunner`](@ref GraphANN.DiskANNRunner).
Since the sample dataset is pretty small, we can get away with using a small search window.
```jldoctest diskann-example; filter = r"0x[0-9a-f]+"
julia> queries = GraphANN.sample_queries()
100-element Vector{StaticArrays.SVector{128, Float32}}:
 [1.0, 3.0, 11.0, 110.0, 62.0, 22.0, 4.0, 0.0, 43.0, 21.0  …  2.0, 2.0, 25.0, 18.0, 8.0, 2.0, 19.0, 42.0, 48.0, 11.0]
 [40.0, 25.0, 11.0, 0.0, 22.0, 31.0, 6.0, 8.0, 10.0, 3.0  …  60.0, 40.0, 4.0, 30.0, 23.0, 32.0, 10.0, 3.0, 19.0, 13.0]
 [28.0, 4.0, 3.0, 6.0, 7.0, 2.0, 2.0, 18.0, 19.0, 37.0  …  59.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 54.0, 47.0]
 [24.0, 12.0, 14.0, 8.0, 3.0, 12.0, 4.0, 8.0, 8.0, 12.0  …  8.0, 0.0, 1.0, 39.0, 15.0, 0.0, 14.0, 103.0, 16.0, 0.0]
 [0.0, 4.0, 47.0, 20.0, 9.0, 2.0, 1.0, 0.0, 41.0, 42.0  …  21.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0, 69.0, 45.0, 2.0]
 [16.0, 52.0, 32.0, 13.0, 2.0, 6.0, 34.0, 49.0, 45.0, 83.0  …  5.0, 6.0, 32.0, 49.0, 21.0, 7.0, 5.0, 0.0, 5.0, 8.0]
 [21.0, 17.0, 52.0, 17.0, 17.0, 10.0, 10.0, 8.0, 25.0, 12.0  …  1.0, 1.0, 21.0, 31.0, 6.0, 8.0, 8.0, 1.0, 2.0, 9.0]
 [3.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 11.0, 15.0  …  4.0, 1.0, 36.0, 20.0, 12.0, 5.0, 11.0, 25.0, 51.0, 15.0]
 [1.0, 12.0, 4.0, 2.0, 41.0, 40.0, 19.0, 5.0, 3.0, 30.0  …  61.0, 18.0, 99.0, 9.0, 8.0, 4.0, 9.0, 6.0, 17.0, 50.0]
 [0.0, 11.0, 77.0, 24.0, 3.0, 0.0, 0.0, 0.0, 28.0, 70.0  …  19.0, 4.0, 0.0, 0.0, 2.0, 1.0, 6.0, 53.0, 33.0, 2.0]
 ⋮
 [5.0, 0.0, 0.0, 2.0, 22.0, 1.0, 0.0, 11.0, 1.0, 6.0  …  131.0, 75.0, 45.0, 14.0, 1.0, 18.0, 59.0, 3.0, 2.0, 11.0]
 [113.0, 12.0, 2.0, 0.0, 0.0, 0.0, 0.0, 37.0, 59.0, 28.0  …  107.0, 101.0, 37.0, 4.0, 5.0, 7.0, 2.0, 0.0, 17.0, 122.0]
 [42.0, 8.0, 0.0, 0.0, 22.0, 56.0, 10.0, 15.0, 133.0, 44.0  …  133.0, 24.0, 3.0, 0.0, 2.0, 2.0, 1.0, 24.0, 37.0, 47.0]
 [113.0, 2.0, 0.0, 0.0, 0.0, 0.0, 8.0, 78.0, 115.0, 3.0  …  125.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 105.0, 20.0]
 [60.0, 70.0, 39.0, 33.0, 26.0, 4.0, 0.0, 0.0, 34.0, 12.0  …  0.0, 0.0, 0.0, 3.0, 121.0, 30.0, 0.0, 0.0, 0.0, 0.0]
 [13.0, 7.0, 4.0, 27.0, 2.0, 0.0, 3.0, 19.0, 91.0, 61.0  …  122.0, 122.0, 0.0, 0.0, 0.0, 10.0, 9.0, 24.0, 43.0, 30.0]
 [14.0, 8.0, 0.0, 0.0, 3.0, 49.0, 22.0, 8.0, 39.0, 37.0  …  113.0, 102.0, 106.0, 50.0, 0.0, 0.0, 3.0, 32.0, 14.0, 15.0]
 [11.0, 48.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 44.0  …  0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 [4.0, 10.0, 5.0, 90.0, 67.0, 6.0, 4.0, 3.0, 117.0, 104.0  …  79.0, 72.0, 105.0, 4.0, 0.0, 0.0, 15.0, 43.0, 0.0, 8.0]

julia> runner = GraphANN.DiskANNRunner(index, 10);

julia> ids = GraphANN.search(runner, index, queries; num_neighbors = 5)
5×100 Matrix{UInt32}:
 0x00000881  0x00000ade  0x00000a94  …  0x0000227a  0x00001555  0x00001f93
 0x00000ea9  0x00002567  0x000026d3     0x0000237a  0x00001540  0x0000224f
 0x00000373  0x000009bd  0x00000a8b     0x000017ff  0x000016b3  0x000012a0
 0x00000faa  0x0000052b  0x000026f5     0x000013d1  0x00001ae3  0x000023cc
 0x00000b16  0x00000c41  0x000022cb     0x0000185b  0x0000168e  0x000026d7
```
Since we asked for the 5 nearest neighbors for each of our 100 queries, the size of the resulting integer matrix is `(5, 100)`.
Here, entry `ids[1, 1]` is the nearest neighbor for `queries[1]` (approximately), `ids[2, 1]` is the second closest neighbor etc.
The function [`recall`](@ref GraphANN.recall) will compute the 5-recall@5 for the returned ids:
```jldoctest diskann-example; filter = r"[0-9]\.[0-9]"
julia> groundtruth = GraphANN.sample_groundtruth();

julia> recalls = GraphANN.recall(groundtruth, ids)
100-element Vector{Float64}:
 0.8
 1.0
 0.8
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 ⋮
 1.0
 1.0
 1.0
 1.0
 1.0
 0.8
 1.0
 1.0
 1.0
```
The average recall can easily be computed.
```jldoctest diskann-example; filter = r"[0-9\.]+"
julia> using Statistics; mean(recalls)
0.984
```

### Multithreaded Query

To use all available threads for querying, construct the `runner` object above with the `executor = GraphANN.dynamic_thread` keyword argument.
```jldoctest diskann-example; filter = r"[0-9\.]+"
julia> runner = GraphANN.DiskANNRunner(index, 10; executor = GraphANN.dynamic_thread);

julia> ids = GraphANN.search(runner, index, queries; num_neighbors = 5);

julia> mean(GraphANN.recall(groundtruth, ids))
0.984
```

## Gathering Telemetry

The [`search`](@ref GraphANN.search(::GrpahANN.DiskANNRunner, ::GrpahANN.DIskANNIndex, ::Any)
methods accept a [`DiskANNCallbacks`](@ref GraphANN.Algorithms.DiskANNCallbacks) struct to help with
gethering metrics during querying. The struct contains several arbitrary functions that are
called at key points during the querying process, allowing for arbitrary telementry to
be defined.

### Latency Measurements

An example of using the callback mechanism to perform latency measurements is shown below.
```jldoctest diskann-example; filter = r"0x[0-9a-f]+"
julia> runner = GraphANN.DiskANNRunner(index, 10; executor = GraphANN.dynamic_thread);

julia> latencies, callbacks = GraphANN.Algorithms.latency_callbacks(runner);

julia> get(latencies) # Initially, latency values are not populated.
UInt64[]

julia> ids = GraphANN.search(runner, index, queries; num_neighbors = 5, callbacks = callbacks);

julia> get(latencies) # Latency in `ns` reported for each query.
100-element Vector{UInt64}:
 0x000000000001cfe1
 0x00000000000130a4
 0x000000000001178a
 0x000000000001353d
 0x000000000001275d
 0x00000000000115bf
 0x0000000000011cbc
 0x0000000000018443
 0x000000000000f11b
 0x0000000000011a49
                  ⋮
 0x0000000000008d8a
 0x0000000000009dca
 0x000000000000aaf9
 0x000000000000a5a6
 0x0000000000011e38
 0x000000000000e311
 0x000000000000aa18
 0x000000000000feaa
 0x0000000000008137
```



## Docstrings
```@docs
GraphANN.DiskANNRunner
GraphANN.DiskANNIndex
GraphANN.build(::Any, ::GraphANN.DiskANNIndexParameters)
GraphANN.search(::GraphANN.DiskANNRunner, ::GraphANN.DiskANNIndex, ::Any)
GraphANN.search(::GraphANN.DiskANNRunner, ::GraphANN.DiskANNIndex, ::AbstractVector{<:AbstractVector})
GraphANN.DiskANNIndexParameters
GraphANN.Algorithms.DiskANNCallbacks
GraphANN.Algorithms.getresults!
```
