# Custom Dataset
# Note the safest data structure in the world, so use with care.
# Type Parameters:
#
# - `S`: Power of 2 that bounds each segment.
#        That is, ALL segments except for the last should have length `2 ^ S`.
# - `T`: Element type stored by the dataset.
# - `A`: Inner vector types.
struct SplitDataset{S,T,A<:AbstractVector{T}} <: AbstractVector{T}
    # N.B. - Don't move ANY of the vecters after their initial allocation.
    # Otherwise, the cached pointers are no longer going to be valid and then you're going
    # to have a really bad day.
    #
    # Accessing the chunks directly should not be accessed directly for this reason ...
    chunks::Vector{A}
    pointers::Vector{Ptr{T}}

    # Total length of the combined chunks.
    length::Int
end

function SplitDataset{S}(chunks::Vector{A}) where {S,T,A<:AbstractVector{T}}
    return SplitDataset{S,T,A}(chunks, pointer.(chunks), sum(length, chunks))
end

Base.size(A::SplitDataset) = (A.length,)
@inline function _index(::Val{S}, i::Integer) where {S}
    mask = (one(i) << S) - one(i)
    _i = i - one(i)
    lo = (_i & mask)
    hi = _i >> S
    return (hi + one(hi), lo)
end

function Base.getindex(A::SplitDataset{S,T}, i::Int) where {S,T}
    @boundscheck checkbounds(A, i)
    return unsafe_load(pointer(A, i))
end

Base.IndexStyle(::Type{<:SplitDataset}) = Base.IndexLinear()
@inline function Base.pointer(A::SplitDataset{S,T}, i::Integer) where {S,T}
    hi, lo = _index(Val(S), i)
    return @inbounds(A.pointers[hi] + lo * sizeof(T))
end

#####
##### Top Constructor
#####

_bytes(::Type{T}, s) where {T} = sizeof(T) * 2^s

"""
    split_partition_size(::Type{T}, fastbytes, slack) -> NamedTuple

Compute the largest partition size (expressed as the exponent for a power of 2) that can
be used to partition a dataset with element type `T` such that an integer number of
partitions can be allocated in fewer than `fastbytes` bytes without losing more than `slack`
bytes.
"""
function split_partition_size(::Type{T}, fastbytes, slack) where {T}
    s = ceil(Int, log2(fastbytes))
    bytes = _bytes(T, s)
    while (fastbytes - bytes * div(fastbytes, bytes)) > slack
        x = fastbytes - bytes * div(fastbytes, bytes)
        s -= 1
        bytes = _bytes(T, s)
    end

    bytes_lost = fastbytes - bytes * div(fastbytes, bytes)
    return (
        partition_size = s,
        num_fast_partitions = div(fastbytes, bytes),
        bytes_lost = bytes_lost,
    )
end

"""
    split(data, partition_size, num_fast_partitions; [fast_allocator], [slow_allocator])

Split the dataset into partitions with `num_fast_partitions` allocated by the
`fast_allocator` and the remainder allocated by the `slow_allocator`.

Each partition will hold `2 ^ partition_size` elements.

Arguments `partition_size` and `num_fast_partitions` should come from the
[`split_partition_size`](@ref) method.
"""
function split(
    data::AbstractVector{T},
    partition_size::Integer,
    num_fast_partitions::Integer;
    fast_allocator = stdallocator,
    slow_allocator = stdallocator,
) where {T}
    chunks = Vector{typeof(data)}()
    fast_partitions_allocated = 0
    for batch in batched(eachindex(data), 2^partition_size)
        if fast_partitions_allocated < num_fast_partitions
            chunk = fast_allocator(T, length(batch))
            fast_partitions_allocated += 1
            println("Fast")
        else
            chunk = slow_allocator(T, length(batch))
            println("Slow")
        end

        # Populate!
        chunk .= view(data, batch)
        push!(chunks, chunk)
    end
    return SplitDataset{partition_size}(chunks)
end
