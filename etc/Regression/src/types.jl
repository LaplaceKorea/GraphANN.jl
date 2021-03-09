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
        close(io)
        mv(path, record.path; force = true)
    end
end

lower(x) = x
lower(x::AbstractDict) = lowerdict(x)
lowerdict(x) = Dict(key => lower(value) for (key, value) in pairs(x))
Base.push!(record::Record, row; cols = :union) = push!(record.df, lowerdict(row); cols = cols)

# Overloads for lowering some GraphANN types
lower(::typeof(GraphANN.stdallocator)) = "DRAM"
lower(allocator::GraphANN._Base.PMAllocator) = "PM - $(allocator.path)"
lower(::Type{GraphANN.Euclidean{N,T}}) where {N,T} = "Euclidean{$N,$T}"

_wrap(x::Union{Tuple,Vector}) = x
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

abstract type LazyLoader end

function exclude(x::LazyLoader)
    return [name for name in propertynames(x) if startswith(String(name), "_")]
end

function cleanup(x::LazyLoader)
    for name in exclude(x)
        setproperty!(x, name, nothing)
    end
end

Base.@kwdef mutable struct Dataset <: LazyLoader
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

#####
##### Graph Loader
#####

Base.@kwdef mutable struct Graph <: LazyLoader
    path::String
    graph_allocator = GraphANN.stdallocator
    # Same story as the Dataset
    _graph::Any = nothing
end

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

#####
##### Clustered Things
#####

abstract type QuantizationDistanceType end
struct EagerDistance <: QuantizationDistanceType end
struct LazyDistance <: QuantizationDistanceType end
lower(x::QuantizationDistanceType) = string(typeof(x))

abstract type QuantizationDistanceStrategy end
struct EncodedData <: QuantizationDistanceStrategy end
struct EncodedGraph <: QuantizationDistanceStrategy end
lower(x::QuantizationDistanceStrategy) = string(typeof(x))

Base.@kwdef mutable struct Quantization <: LazyLoader
    path::String
    num_centroids::Int
    num_partitions::Int

    encoded_data_allocator
    pqgraph_allocator

    # -- Cached Items
    _pqtable = nothing
    _pqtranspose = nothing
    _encoder = nothing
    _data_encoded = nothing
    _pqgraph = nothing
end
# Lazy Loaders
function _pqtable(q::Quantization)
    if q._pqtable === nothing
        centroids = deserialize(q.path)
        @assert size(centroids, 1) == q.num_centroids
        @assert size(centroids, 2) == q.num_partitions
        q._pqtable = GraphANN.PQTable{size(centroids, 2)}(centroids)
    end
    return q._pqtable
end

function  _pqtranspose(q::Quantization)
    if q._pqtranspose === nothing
        pqtable = _pqtable(q)
        q._pqtranspose = GraphANN._Quantization.PQTransposed(pqtable)
    end
    return q._pqtranspose
end

function _encoder(q::Quantization)
    if q._encoder === nothing
        pqtable = _pqtable(q)
        q._encoder = GraphANN._Quantization.binned(q._pqtable)
    end
    return q._encoder
end

function _data_encoded(q::Quantization, dataset::Dataset)
    if q._data_encoded === nothing
        qtype = q.num_centroids <= 256 ? UInt8 : UInt16
        encoder = _encoder(q)
        q._data_encoded = GraphANN._Quantization.encode(
            qtype,
            encoder,
            load(dataset);
            allocator = q.encoded_data_allocator,
        )
    end
    return q._data_encoded
end

function getmeta(q::Quantization, ::EncodedData, dataset::Dataset, graph::Graph)
    return GraphANN.MetaGraph(load(graph), _data_encoded(q, dataset))
end

function getmeta(q::Quantization, ::EncodedGraph, dataset::Dataset, graph::Graph)
    if q._pqgraph === nothing
        qtype = q.num_centroids <= 256 ? UInt8 : UInt16
        encoder = _encoder(q)
        q._pqgraph = GraphANN.PQGraph{qtype}(
            encoder,
            load(graph),
            load(dataset);
            allocator = q.pqgraph_allocator,
        )
    end
    return GraphANN.MetaGraph(load(graph), q._pqgraph)
end

getmetric(q::Quantization, ::EagerDistance) = _pqtranspose(q)
getmetric(q::Quantization, ::LazyDistance) = _pqtable(q)

