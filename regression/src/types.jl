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

