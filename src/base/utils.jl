# get rid of that pesky "can't reduce over empty collection" error.
safe_maximum(f::F, itr, default = 0) where {F} = isempty(itr) ? default : maximum(f, itr)
donothing(x...; kw...) = nothing
always_false(x...; kw...) = false
printlnstyled(x...; kw...) = printstyled(x..., "\n"; kw...)

# Convenience definitions
zero!(x) = fill!(x, zero(eltype(x)))
typemax!(x) = fill!(x, typemax(eltype(x)))

# Ceiling division
cdiv(a::Integer, b::Integer) = cdiv(promote(a, b)...)
cdiv(a::T, b::T) where {T<:Integer} = one(T) + div(a - one(T), b)

toeltype(::Type{T}, x::AbstractArray) where {T} = map(i -> convert(T, i), x)
function toeltype(::Type{T}, x::AbstractArray{<:AbstractArray}) where {T}
    return map(i -> toeltype(T, i), x)
end

clog2(x::T) where {T<:Integer} = ceil(T, log2(x))

#####
##### Neighbor
#####

# NOTE: Don't define conversion functions like:
# `convert(Neighbor{Float32}, ::Neighbor{Int32})`.
# If this function gets called, it means some data structure wasn't initialized right.
# We don't want this kind of implicit conversion to happen since it will usually result in
# less than optimal code.

"""
    Neighbor

Lightweight struct for containing an ID/distance pair with a total ordering.
"""
struct Neighbor{T,D}
    id::T
    distance::D

    # Inner conversion constructors
    # Require explicitly calling out integer type.
    Neighbor{T}(id, distance::D) where {T,D} = Neighbor{T,D}(id, distance)
    Neighbor{T,D}(id, distance::D) where {T,D} = new{T,D}(convert(T, id), distance)
    Neighbor{T,D}() where {T,D} = new{T,D}(zero(T), typemax(D))
end

# Since `Neighbor` contains two generic fields (id and distance), we need to provide hooks
# to allow users of `Neighbor` to preallocate types with the correct parameters.
# These are `idtype` and `costtype` respectively.
idtype(::Type{T}) where {T} = T
idtype(::T) where {T} = idtype(T)

# For convenience, define two-arg and one-arg versions to allow for promotion.
#
# N.B. This relies on Julia's type inference mechanism to determine the return type of
# calling `evaluate`.
#
# Relying on inference is generally discouraged, so tread lightly.
"""
    costtype(metric, type1, type2)

Return the result type of applying evaluating `metric` on the two types.

# Example
```julia
julia> GraphANN.costtype(GraphANN.Euclidean(), GraphANN.SVector{32,UInt8}, GraphANN.SVector{32,UInt8})
Int32
```
"""
function costtype(::Metric, ::Type{A}, ::Type{B}) where {Metric,A,B}
    return Base.promote_op(evaluate, Metric, A, B)
end

costtype(metric, ::Type{A}) where {A} = costtype(metric, A, A)
costtype(metric, ::A) where {A} = costtype(metric, A, A)
costtype(metric, ::A, ::B) where {A,B} = costtype(metric, A, B)
costtype(metric, ::AbstractVector{T}) where {T} = costtype(metric, T, T)

# Define `getid` for integers as well so we can use `getid` in all places where we need
# an id without fear.
getid(x::Integer) = x
getid(x::Neighbor) = x.id
getdistance(x::Neighbor) = x.distance

# Implement a total ordering on Neighbors
# This is important for keeping queues straight.
# This is tested and you WILL be found if you break it :)
function Base.isless(a::Neighbor, b::Neighbor)
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id)
end

# Convenience for array indexing
@inline Base.getindex(A::AbstractArray, i::Neighbor) = A[getid(i)]
@inline Base.pointer(A::AbstractArray, i::Neighbor) = pointer(A, getid(i))

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

# Slightly hijack internals of "sets" to perform a function call only if a key doesn't
# exist (and add the key to the set).
@inline function ifmissing!(f::F, set::AbstractSet, key) where {F}
    return get!(() -> (f(); nothing), set.dict, key)
end

#####
##### Nearest Neighbor
#####

"""
    nearest_neighbor(query, dataset; [metric])

Return the brute-force nearest neighbor to `query` in `dataset`.
Takes an arbitrary `metric`.

!!! warning

    Function is multithreaded. Do not call with multiple threads.
"""
function nearest_neighbor(query::T, data::AbstractVector; metric = Euclidean()) where {T}
    # Thread local storage is a NamedTuple with the following fields:
    # `min_ind` - The index of the nearest neighbor seen by this thread so far.
    # `min_dist` - The distance corresponding to the current nearest neighbor.
    tls = ThreadLocal((min_ind = 0, min_dist = typemax(eltype(query))))

    dynamic_thread(1:length(data), 128) do i
        @inbounds x = data[i]
        @unpack min_ind, min_dist = tls[]
        dist = evaluate(metric, query, x)

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
##### Medoid
#####

"""
    medioid(dataset) -> Int64

Return the index of the element in `dataset` that is nearest to the true `medioid` of the
dataset.

The `medioid` is defined as the element-wise mean of all items in the dataset.
"""
function medioid(data::AbstractVector{SVector{N,T}}) where {N,T}
    # Thread to make fast for larger datasets.
    # Also, use floating-point for accumulation to avoid overflow errors.
    # It's not perfect, but probably okay.
    tls = ThreadLocal(zero(SVector{N,Float32}))
    dynamic_thread(data, 1024) do i
        tls[] += i
    end
    medioid = sum(getall(tls)) / length(data)
    return first(nearest_neighbor(medioid, data))
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
"""
    recall(groundtruth::AbstractMatrix, ids::AbstractMatrix) -> Float64

Return the average `N` recall@`N` where `N = size(ids, 1)`.

Expects `groundtruth` and `ids` to be integer matrices with nearest neighbor indices
grouped along columns. For example, `groundtruth[1:5, 10]` returns the sorted 5 nearest
neighbors for query 10 (whatever that may be).

Note - it's okay for `size(groundtruth, 1) >= size(groundtruth, 2)`.
"""
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
        Base.llvmcall(
            (
                raw"""
     define void @entry(i64) #0 {
     top:
         %val = inttoptr i64 %0 to i8*
         call void asm sideeffect "prefetch $0", "*m,~{dirflag},~{fpsr},~{flags}"(i8* nonnull %val)
         ret void
     }

     attributes #0 = { alwaysinline }
     """,
                "entry",
            ),
            Cvoid,
            Tuple{Ptr{Cvoid}},
            Ptr{Cvoid}(ptr),
        )
        return nothing
    end

    function prefetch_llc(ptr::Ptr)
        Base.@_inline_meta
        Base.llvmcall(
            (
                raw"""
     define void @entry(i64) #0 {
     top:
         %val = inttoptr i64 %0 to i8*
         call void asm sideeffect "prefetcht1 $0", "*m,~{dirflag},~{fpsr},~{flags}"(i8* nonnull %val)
         ret void
     }

     attributes #0 = { alwaysinline }
     """,
                "entry",
            ),
            Cvoid,
            Tuple{Ptr{Cvoid}},
            Ptr{Cvoid}(ptr),
        )
        return nothing
    end
else
    function prefetch(ptr::Ptr)
        Base.@_inline_meta
        Base.llvmcall(
            raw"""
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
        Base.llvmcall(
            raw"""
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

function prefetch(
    A::AbstractVector{T}, i, f::F = _Base.prefetch
) where {T<:AbstractVector,F}
    # Need to prefetch the entire vector
    # Compute how many cache lines are needed.
    # Divide the number of bytes by 64 to get cache lines.
    cache_lines = sizeof(T) >> 6
    ptr = pointer(A, i)
    for j in Base.OneTo(cache_lines)
        f(ptr + 64 * (j - 1))
    end
    return nothing
end

function unsafe_prefetch(A::AbstractVector{T}, f::F = _Base.prefetch) where {T <: Number, F}
    cache_lines = length(A) >> 6
    ptr = pointer(A)
    for j in Base.OneTo(cache_lines)
        f(ptr + 64 * (j - 1))
    end
end

#####
##### BatchedRange
#####

struct BatchedRange{T<:AbstractRange}
    range::T
    batchsize::Int64
end

Base.length(x::BatchedRange) = ceil(Int, length(x.range) / x.batchsize)
Base.lastindex(x::BatchedRange) = length(x)

Base.eltype(x::BatchedRange{T}) where {T} = _replace_oneto(T)
_replace_oneto(::Type{T}) where {T<:AbstractRange} = T
_replace_oneto(::Type{Base.OneTo{I}}) where {I} = UnitRange{I}

"""
    batched(range::AbstractRange, batchsize) -> itr

Return an indexable iterator that lazily partitions `range` into subranges with length
`batchsize`.

# Example
```julia
julia> x = 1:10
1:10

julia> b = GraphANN.batched(x, 3);

julia> collect(b)
4-element Vector{UnitRange{Int64}}:
 1:3
 4:6
 7:9
 10:10

julia> b[2]
4:6

julia b[end]
10:10
```
"""
function batched(range::AbstractRange, batchsize::Integer)
    return BatchedRange(range, convert(Int64, batchsize))
end

Base.@propagate_inbounds function Base.getindex(x::BatchedRange, i::Integer)
    @unpack range, batchsize = x
    start = batchsize * (i - 1) + 1
    stop = min(length(range), batchsize * i)
    return subrange(range, start, stop)
end

Base.iterate(x::BatchedRange, s = 1) = s > length(x) ? nothing : (x[s], s + 1)

# handle unit ranges and more general ranges separately so a `BatchedRange` returns
# a `UnitRange` when it encloses a `UnitRange`.
subrange(range::OrdinalRange, start, stop) = range[start]:step(range):range[stop]
subrange(range::AbstractUnitRange, start, stop) = range[start]:range[stop]

#####
##### Bounded Heap
#####

struct Keeper{T,O<:Base.Ordering}
    heap::DataStructures.BinaryHeap{T,O}
    bound::Int

    function Keeper{T}(ordering::Base.Ordering, bound::Integer) where {T}
        # pre-allocate space for the underlying vector.
        vector = Vector{T}(undef, bound + 1)
        empty!(vector)
        heap = DataStructures.BinaryHeap{T}(ordering, vector)
        return new{T,typeof(ordering)}(heap, convert(Int, bound))
    end
end

const KeepLargest{T} = Keeper{T,Base.ForwardOrdering}
const KeepSmallest{T} = Keeper{T,Base.ReverseOrdering{Base.ForwardOrdering}}

KeepLargest{T}(bound::Integer) where {T} = Keeper{T}(Base.ForwardOrdering(), bound)
KeepSmallest{T}(bound::Integer) where {T} = Keeper{T}(Base.ReverseOrdering(), bound)
isfull(H::Keeper) = length(H) >= H.bound

# slight type hijacking ...
# we don't technically "own" valtree.
Base.empty!(H::Keeper) = empty!(H.heap.valtree)
Base.isempty(H::Keeper) = isempty(H.heap)
Base.length(H::Keeper) = length(H.heap)
Base.first(H::Keeper) = first(H.heap)
getbound(H::Keeper) = H.bound
getordering(H::Keeper) = H.heap.ordering

"""
    push!(H::Keeper, i)

Add `i` to `H` if (1) `H` is not full or (2) `i` is less than maximal element in `H`.
After calling `push!`, the length of `H` will be less than or equal to its established
bound.
"""
function Base.push!(H::Keeper, i)
    o = getordering(H)
    if (length(H.heap) < H.bound || Base.lt(o, first(H.heap), i))
        push!(H.heap, i)

        # Wrap in a "while" loop to handle the case where extra things got added somehow.
        # (i.e., someone used this type incorrectly)
        while length(H.heap) > H.bound
            pop!(H.heap)
        end
    end
    return nothing
end

function destructive_extract!(H::Keeper)
    sort!(
        H.heap.valtree; alg = Base.QuickSort, order = Base.ReverseOrdering(getordering(H))
    )
    return H.heap.valtree
end

#####
##### Compute mean and variance online
#####

# This is closely modeled on `_var(iterable, corrected::Bool, mean)` in the Statistics
# `stdlib`, but slightly modified to take a default return value in the case of an empty
# iterable rather than relying on the iterable implementing `eltype`.
# N.B. Inference can be finicky with this function. Use with care.
function meanvar(::Type{T}, iterable, corrected::Bool = true) where {T}
    y = iterate(iterable)
    if y === nothing
        return (mean = zero(T), variance = zero(T))
    end
    count = 1
    value, state = y
    y = iterate(iterable, state)
    M = convert(T, value)
    S = zero(M)
    while y !== nothing
        value, state = y
        valueT = convert(T, value)
        y = iterate(iterable, state)
        count += 1
        new_M = M + (valueT - M) / count
        S = S + ziptimes(valueT - M, valueT - new_M)
        M = new_M
    end
    return (mean = M, variance = S / (count - Int(corrected)))
end

ziptimes(a::Number, b::Number) = a * b
ziptimes(a::AbstractVector, b::AbstractVector) = a .* b

