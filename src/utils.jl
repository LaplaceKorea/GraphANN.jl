#####
##### Neighbor
#####

struct Neighbor
    id::Int
    distance::Float64
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
end

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
    medioid = T(mapreduce(raw, (x,y) -> x .+ y, data) ./ length(data))
    return first(nearest_neighbor(medioid, data))
end

#####
##### nearest_neighbor
#####

function nearest_neighbor(query::T, data::Vector{T}) where {T}
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

