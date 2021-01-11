module Regression

# You know - the thing we're actually testing.
import GraphANN

# stdlib
using Serialization
using Statistics

# deps
using DataFrames
using DataStructures

import PrettyTables
import UnPack: @unpack, @pack!

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const SCRATCH = joinpath(SRCDIR, "data")

makescratch() = ispath(SCRATCH) || mkpath(SCRATCH)

include("test.jl")

#####
##### Implementation
#####

# Dataframe Wrapper
struct Record
    df::DataFrame
    path::String
end

function Record(path::AbstractString, new = false)
    df = (new == false && ispath(path)) ? (deserialize(path)::DataFrame) : (DataFrame())
    return Record(df, path)
end

_transpose(df) = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
Base.show(io::IO, record::Record) = show(io, _transpose(record.df))

function save(record::Record)
    mktemp() do path, io
        serialize(io, record.df)
        mv(path, record.path; force = true)
    end
end

lower(x) = x
function Base.push!(record::Record, row; cols = :union)
    row = Dict(key => lower(value) for (key, value) in pairs(row))
    return push!(record.df, row; cols = cols)
end

# Overloads for lowering some GraphANN types
lower(::typeof(GraphANN.stdallocator)) = "DRAM"
lower(allocator::GraphANN._Base.PMAllocator) = "PM - $(allocator.path)"
lower(::Type{GraphANN.Euclidean{N,T}}) where {N,T} = "Euclidean{$N,$T}"

_wrap(x::Tuple) = x
_wrap(x::Symbol) = (x,)
exclude(::Any) = ()
function dict(x::T; excluded = exclude(x)) where {T}
    names = propertynames(x)
    return Dict(n => getproperty(x, n) for n in names if !in(n, _wrap(excluded)))
end

makeresult(v::AbstractVector) = SortedDict(reduce(merge, v))

#####
##### Experiment Data Types
#####

Base.@kwdef mutable struct Dataset
    path::String
    eltype::Type{<:GraphANN.Euclidean}
    maxlines::Union{Int,Nothing} = nothing
    data_allocator = GraphANN.stdallocator
    # Other related items.
    groundtruth::String
    queries::String
    # Memoize the dataset since it can sometime be expensive to load.
    _dataset::Any = nothing
end
exclude(::Dataset) = (:_dataset,)

function load(dataset::Dataset)
    @unpack _dataset = dataset
    _dataset !== nothing && return _dataset

    @unpack path, maxlines, data_allocator, eltype = dataset
    _dataset =  GraphANN.load_vecs(
        eltype,
        path;
        maxlines = maxlines,
        allocator = data_allocator
    )
    @pack! dataset = _dataset
    return _dataset
end

function load_queries(dataset::Dataset)
    @unpack queries, eltype = dataset
    return GraphANN.load_vecs(eltype, queries)
end

# Add 1 to convert from `index-0` to `index-1`.
load_groundtruth(dataset::Dataset) = GraphANN.load_vecs(dataset.groundtruth) .+ 1
function loadall(x::Dataset)
    return (data = load(x), queries = load_queries(x), groundtruth = load_groundtruth(x))
end
cleanup(x::Dataset) = (x._dataset = nothing)

Base.@kwdef mutable struct Graph
    path::String
    graph_allocator = GraphANN.stdallocator
    # Same story as the Dataset
    _graph::Any = nothing
end
exclude(::Graph) = (:_graph,)

function load(graph::Graph)
    @unpack _graph = graph
    _graph !== nothing && return _graph

    @unpack path, graph_allocator = graph
    _graph = GraphANN.load(
        GraphANN.DenseAdjacencyList{UInt32},
        path;
        allocator = graph_allocator
    )
    @pack! graph = _graph
    return _graph
end
cleanup(x::Graph) = (x._graph = nothing)

# -- Index Building top level
function index_building(
    record::Record,
    dataset::Dataset,
    parameters::GraphANN.GraphParameters;
    savepath = nothing,
    batchsize = 50000,
    allocator = GraphANN.stdallocator,
)
    data = load(dataset)
    stats = @timed meta = GraphANN.generate_index(data, parameters; batchsize = batchsize)

    if savepath != nothing
        GraphANN.save(savepath, meta.graph)
    end

    # Build up the things we want to save
    results = makeresult([
        dict(stats; excluded = :value),
        dict(dataset),
        dict(parameters),
        Dict(
            :batchsize => batchsize,
            :savepath => savepath,
            :num_threads => Threads.nthreads(),
            :graph_allocator => allocator,
        )
    ])

    # Save the results!
    push!(record, results)
    save(record)
    return nothing
end

#####
##### Queries
#####

struct Memoize{F}
    f::F
    saved::Dict{Any,Any}
end

memoize(f::F) where {F} = Memoize{F}(f, Dict{Any,Any}())
(f::Memoize)(x...) = get!(() -> f.f(x...), f.saved, x)

# There are ... a lot of options for parameters running during queries.
# I'm going to try to parameterize these as types and we'll see how well that works out.
abstract type AbstractQueryParameter end
lower(::T) where {T <: AbstractQueryParameter} = string(T)

abstract type AbstractCallbacks <: AbstractQueryParameter end
struct NoCallbacks <: AbstractCallbacks end
struct LatencyCallbacks <: AbstractCallbacks end

abstract type AbstractThreading <: AbstractQueryParameter end
struct SingleThread <: AbstractThreading end
struct MultiThread <: AbstractThreading end

abstract type AbstractPrefetching <: AbstractQueryParameter end
struct NoPrefetching <: AbstractPrefetching end
struct WithPrefetching <: AbstractPrefetching end

# Dispatch Rules - Callbacks
get_callbacks(::NoCallbacks, ::Any) = (callbacks = GraphANN.GreedyCallbacks(),)
get_callbacks(::LatencyCallbacks, ::SingleThread) = GraphANN.latency_callbacks()
get_callbacks(::LatencyCallbacks, ::MultiThread) = GraphANN.latency_mt_callbacks()

# Unpack the names tuple to reset the latencies Vector or ThreadLocal
reset!(x::NamedTuple, ::NoCallbacks) = nothing
reset!(x::NamedTuple, cb::LatencyCallbacks) = reset!(x.latencies, cb)
reset!(x::AbstractVector, ::LatencyCallbacks) = empty!(x)
reset!(x::GraphANN.ThreadLocal, ::LatencyCallbacks) = empty!.(GraphANN.getall(x))

cb_stats(::NamedTuple, ::NoCallbacks) = NamedTuple()
cb_stats(x::NamedTuple, cb::LatencyCallbacks) = cb_stats(x.latencies, cb)
function cb_stats(x::AbstractVector, ::LatencyCallbacks)
    return (
        mean_latency = mean(x),
        max_latency = maximum(x),
        min_latency = minimum(x),
        nines_latency = nines(x),
    )
end
function cb_stats(x::GraphANN.ThreadLocal, cb::LatencyCallbacks)
    x = reduce(vcat, GraphANN.getall(x))
    return cb_stats(x, cb)
end

# Dispatch Rules - Algorithm Construction
function make_algo(windowsize::Integer, ::SingleThread, ::NoPrefetching, x...)
    return GraphANN.GreedySearch(windowsize)
end

function make_algo(windowsize::Integer, ::MultiThread, ::NoPrefetching, x...)
    return GraphANN.ThreadLocal(GraphANN.GreedySearch(windowsize))
end

# -- Query top level
function query(
    record::Record,
    dataset::Dataset,
    graph::Graph;
    target_accuracies = [0.95, 0.98, 0.99],
    num_neighbors = 1,
    # Number of times to run queries in a row.
    num_loops = 5,
    # higher level parameters.
    callbacks::AbstractCallbacks = NoCallbacks(),
    threading::AbstractThreading = SingleThread(),
    prefetching::AbstractPrefetching = NoPrefetching(),
    maxwindow = 400,
)
    data, queries, groundtruth = loadall(dataset)
    start_index = GraphANN.medioid(data)
    start = GraphANN.StartNode(start_index, data[start_index])
    meta = GraphANN.MetaGraph(load(graph), data)

    # To find the correct window sizes for the provided accuracies, we create a closure
    # to run the query algorithm and memoize it to make things a little faster.
    closure = function(windowsize::Integer)
        # Even though we might ask for single threaded latency, here we use multithreading
        # just to make convergencde a little faster.
        algo = make_algo(windowsize, MultiThread(), NoPrefetching())
        ids = GraphANN.searchall(algo, meta, start, queries; num_neighbors = num_neighbors)
        return mean(GraphANN.recall(groundtruth, ids))
    end
    memoized = memoize(closure)

    windowsizes = map(target_accuracies) do accuracy
        binarysearch(memoized, accuracy, 1, maxwindow)
    end

    # Now, unpack the callbacks
    callback_tuple = get_callbacks(callbacks, threading)

    for (target, windowsize) in zip(target_accuracies, windowsizes)
        algo = make_algo(windowsize, threading, prefetching)

        # Warmup Round
        f = () -> GraphANN.searchall(
            algo,
            meta,
            start,
            queries;
            num_neighbors = num_neighbors,
            callbacks = callback_tuple.callbacks,
        )
        reset!(callback_tuple, callbacks)

        ids = f()
        recall = mean(GraphANN.recall(groundtruth, ids))

        # The query run!
        # Run the `_repeat` function once to force precompilaion.
        _repeat(f, num_loops)
        reset!(callback_tuple, callbacks)
        stats = @timed _repeat(f, num_loops)
        qps = (num_loops * length(queries)) / stats.time

        # Make an entry in the record
        results = makeresult([
            dict(stats; excluded = :value),
            dict(dataset),
            dict(graph),
            dict(cb_stats(callback_tuple, callbacks)),
            Dict(
                :target_recall => target,
                :num_neighbors => num_neighbors,
                :windowsize => windowsize,
                :recall => recall,
                :callbacks => callbacks,
                :threading => threading,
                :prefetching => prefetching,
                :maxwindow => maxwindow,
                :start_index => start_index,
                :qps => qps,
                :num_loops => num_loops,
                :num_queries => length(queries),
                :num_threads => Threads.nthreads(),
                :runtime => stats.time,
            ),
        ])
        push!(record, results)
        save(record)
    end
end

_repeat(f::F, n) where {F} = [f() for _ in 1:n]

# Can we find the first `x` where `f(x) > target` but `f(x-1) < target`.
function binarysearch(f::F, target, lo::T, hi::T) where {F, T <: Integer}
    u = T(1)
    lo = lo - u
    hi = hi + u
    @inbounds while lo < hi - u
        m = Base.Sort.midpoint(lo, hi)
        @show m f(m) target
        if isless(f(m), target)
            lo = m
        else
            hi = m
        end
    end
    return hi
end

function nines(x; max = 5, rev = false)
    x = sort(x; rev = rev, alg = QuickSort)
    vals = eltype(x)[]
    for pow in 1:max
        frac = 1.0 - (0.1)^pow
        ind = ceil(Int, frac * length(x))

        ind >= length(x) && break
        push!(vals, x[ind])
    end
    @show Int.(vals)
    return vals
end

end # module
