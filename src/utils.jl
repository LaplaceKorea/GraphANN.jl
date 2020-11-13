#####
##### Neighbor
#####

struct Neighbor
    id::UInt32
    distance::Float32
end

getid(x::Int) = x
getid(x::Neighbor) = x.id

getdistance(x::Neighbor) = x.distance

idequal(a::Neighbor, b::Neighbor) = (getid(a) == getid(b))

# Implement a total ordering on Neighbors
# This is important for keeping queues straight.
# This is tested and you WILL be found if you break it :)
function Base.isless(a::Neighbor, b::Neighbor)
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id)
end

# Convenience for array indexing
@inline Base.getindex(A::AbstractArray, i::Neighbor) = A[getid(i)]

#####
##### Robin Set
#####

# Construct a RobinSet by using a thin wrapper around a RobinDict
#
# Implementation roughly based on Base's implementation of Set.
struct RobinSet{T} <: AbstractSet{T}
    dict::DataStructures.RobinDict{T,Nothing}

    # Inner Constructors
    RobinSet{T}() where {T} = new(Dict{T,Nothing}())
    RobinSet(dict::DataStructures.RobinDict{T,Nothing}) where {T} = new{T}(dict)
end

RobinSet(itr) = RobinSet(DataStructures.RobinDict(i => nothing for i in itr))

Base.push!(set::RobinSet, k) = (set.dict[k] = nothing)
Base.pop!(set::RobinSet) = first(pop!(set.dict))
Base.in(k, set::RobinSet) = haskey(set.dict, k)
Base.delete!(set::RobinSet, k) = delete!(set.dict, k)
Base.empty!(set::RobinSet) = empty!(set.dict)
Base.length(set::RobinSet) = length(set.dict)

# Iterator interface
Base.iterate(set::RobinSet) = iterate(keys(set.dict))
Base.iterate(set::RobinSet, s) = iterate(keys(set.dict), s)


#####
##### Medoid
#####

# Find the medioid of a dataset
function medioid(data::Vector{T}) where {T}
    # First, find the element wise sum
    medioid = Euclidean(mapreduce(raw, (x,y) -> x .+ y, data) ./ length(data))
    return first(nearest_neighbor(medioid, data))
end

#####
##### nearest_neighbor
#####

function nearest_neighbor(query::T, data::AbstractVector) where {T}
    min_ind = 0
    min_dist = typemax(eltype(query))
    for (i, x) in enumerate(data)
        dist = distance(query, x)
        if dist < min_dist
            min_dist = dist
            min_ind = i
        end
    end
    return (min_ind, min_dist)
end

#####
##### Compute Recall
#####

function recall(groundtruth::AbstractVector, results::AbstractVector)
    count = 0
    for i in groundtruth
        in(i, results) && (count += 1)
    end
    return count / length(groundtruth)
end

function recall(groundtruth::AbstractMatrix, results::AbstractMatrix)
    @assert size(groundtruth, 1) >= size(results, 1)
    @assert size(groundtruth, 2) == size(results, 2)

    # Slice the ground truth to match the size of the results.
    vgt = view(groundtruth, 1:size(results, 1), :)
    return [recall(_gt, _r) for (_gt, _r) in zip(eachcol(vgt), eachcol(results))]
end

#####
##### Thread Local
#####

# Thread local storage.
struct ThreadLocal{T}
    values::Vector{T}

    # Inner constructor to resolve ambiguities
    ThreadLocal{T}(values::Vector{T}) where {T} = new{T}(values)
end

# Convenience, wrap around a NamedTuple
ThreadLocal(; kw...) = ThreadLocal((;kw...,))

function ThreadLocal(values::T) where {T}
    return ThreadLocal{T}([deepcopy(values) for _ in 1:Threads.nthreads()])
end

Base.getindex(t::ThreadLocal) = t.values[Threads.threadid()]
getall(t::ThreadLocal) = t.values

allthreads() = 1:Threads.nthreads()
