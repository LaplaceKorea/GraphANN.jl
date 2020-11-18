# get rid of that pesky "can't reduce over empty collection" error.
safe_maximum(f::F, itr, default = 0) where {F} = isempty(itr) ? default : maximum(f, itr)

#####
##### Neighbor
#####

"""
    Neighbor

Lightweight struct for containing an ID/distance pair with a total ordering.
"""
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
    medioid = Euclidean(mapreduce(raw, (x,y) -> Float32.(x) .+ Float32.(y), data) ./ length(data))
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
    @assert length(groundtruth) == length(results)

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
##### Lowest Level Prefetch
#####

function prefetch(ptr::Ptr)
    Base.@_inline_meta
    Base.llvmcall(raw"""
        %val = inttoptr i64 %0 to i8*
        call void asm sideeffect "prefetch $0", "*m,~{dirflag},~{fpsr},~{flags}"(i8* nonnull %val)
        ret void
        """,
        Cvoid,
        Tuple{Ptr{Cvoid}},
        Ptr{Cvoid}(ptr),
    )
    return nothing
end

