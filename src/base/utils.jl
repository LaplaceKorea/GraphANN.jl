# get rid of that pesky "can't reduce over empty collection" error.
safe_maximum(f::F, itr, default = 0) where {F} = isempty(itr) ? default : maximum(f, itr)
donothing(x...) = nothing
printlnstyled(x...; kw...) = printstyled(x..., "\n"; kw...)
zero!(x) = (x .= zero(eltype(x)))
typemax!(x) = (x .= typemax(eltype(x)))

# Ceiling division
cdiv(a::Integer, b::Integer) = cdiv(promote(a, b)...)
cdiv(a::T, b::T) where {T <: Integer} = one(T) + div(a - one(T), b)

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

getid(x::Integer) = x
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
zeroas(::Type{T}, x::Number) where {T} = zero(T)

function medioid(data::Vector{T}) where {T}
    # Thread to make fast for larger datasets.
    tls = ThreadLocal(zeroas(Float32, T))
    dynamic_thread(data, 1024) do i
        tls[] += i
    end
    medioid = sum(getall(tls)) / length(data)
    return first(nearest_neighbor(medioid, data))
end

#####
##### nearest_neighbor
#####

function nearest_neighbor(query::T, data::AbstractVector) where {T}
    # Thread local storage is a NamedTuple with the following fields:
    # `min_ind` - The index of the nearest neighbor seen by this thread so far.
    # `min_dist` - The distance corresponding to the current nearest neighbor.
    tls = ThreadLocal((min_ind = 0, min_dist = typemax(eltype(query))))

    dynamic_thread(1:length(data), 128) do i
        @inbounds x = data[i]
        @unpack min_ind, min_dist = tls[]
        dist = distance(query, x)

        # Update Thread Local Storage is closer
        if dist < min_dist
            tls[] = (min_ind = i, min_dist = dist)
        end
    end

    # Find the nearest neighbor across all threads.
    candidates = getall(tls)
    _, i = findmin([x.min_dist for x in candidates])
    return candidates[i]
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

# Convenience function if we have more ground-truth vectors than results.
# Truncates `groundtruth` to have as many neighbors as `results`.
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

# prefetch - into L1?
# prefetch1 - into L2
# prefetch2 - into LLC

# NOTE: The convention for LLVMCALL changes in Version 1.6
@static if VERSION >= v"1.6.0-beta1"
    function prefetch(ptr::Ptr)
        Base.@_inline_meta
        Base.llvmcall((raw"""
            define void @entry(i64) #0 {
            top:
                %val = inttoptr i64 %0 to i8*
                call void asm sideeffect "prefetch $0", "*m,~{dirflag},~{fpsr},~{flags}"(i8* nonnull %val)
                ret void
            }

            attributes #0 = { alwaysinline }
            """, "entry"),
            Cvoid,
            Tuple{Ptr{Cvoid}},
            Ptr{Cvoid}(ptr),
        )
        return nothing
    end

    function prefetch_llc(ptr::Ptr)
        Base.@_inline_meta
        Base.llvmcall((raw"""
            define void @entry(i64) #0 {
            top:
                %val = inttoptr i64 %0 to i8*
                call void asm sideeffect "prefetcht1 $0", "*m,~{dirflag},~{fpsr},~{flags}"(i8* nonnull %val)
                ret void
            }

            attributes #0 = { alwaysinline }
            """, "entry"),
            Cvoid,
            Tuple{Ptr{Cvoid}},
            Ptr{Cvoid}(ptr),
        )
        return nothing
    end
else
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

    function prefetch_llc(ptr::Ptr)
        Base.@_inline_meta
        Base.llvmcall(raw"""
            %val = inttoptr i64 %0 to i8*
            call void asm sideeffect "prefetcht1 $0", "*m,~{dirflag},~{fpsr},~{flags}"(i8* nonnull %val)
            ret void
            """,
            Cvoid,
            Tuple{Ptr{Cvoid}},
            Ptr{Cvoid}(ptr),
        )
        return nothing
    end
end

function unsafe_prefetch(x::AbstractVector{T}, i, len) where {T}
    # Compute the number of cache lines accessed.
    num_cache_lines = ceil(Int, len * sizeof(T) / 64)
    ptr = pointer(x, i)
    for j in 0:(num_cache_lines - 1)
        prefetch(ptr + 64 * j)
    end
end

function prefetch(A::AbstractVector{T}, i, f::F = _Base.prefetch) where {T,F}
    # Need to prefetch the entire vector
    # Compute how many cache lines are needed.
    # Divide the number of bytes by 64 to get cache lines.
    cache_lines = sizeof(T) >> 6
    ptr = pointer(A, i)
    for i in 1:cache_lines
        f(ptr + 64 * (i-1))
    end
    return nothing
end


#####
##### BatchedRange
#####

struct BatchedRange{T <: AbstractRange}
    range::T
    batchsize::Int64
end

Base.length(x::BatchedRange) = ceil(Int, length(x.range) / x.batchsize)

function batched(range::AbstractRange, batchsize::Integer)
    return BatchedRange(range, convert(Int64, batchsize))
end

Base.@propagate_inbounds function Base.getindex(x::BatchedRange, i::Integer)
    @unpack range, batchsize = x
    start = batchsize * (i-1) + 1
    stop = min(length(range), batchsize * i)
    return subrange(range, start, stop)
end

Base.iterate(x::BatchedRange, s = 1)  = s > length(x) ? nothing : (x[s], s+1)

# handle unit ranges and more general ranges separately so a `BatchedRange` returns
# a `UnitRange` when it encloses a `UnitRange`.
subrange(range::OrdinalRange, start, stop) = range[start]:step(range):range[stop]
subrange(range::AbstractUnitRange, start, stop) = range[start]:range[stop]

#####
##### Debug Utils
#####

export @ttime
macro ttime(expr)
    expr = esc(expr)
    return quote
        if Threads.threadid() == 1
            @time $expr
        else
            $expr
        end
    end
end
