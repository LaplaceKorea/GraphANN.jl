# Custom Dataset
# Note the safest data structure in the world, so use with care.
struct SplitDataset{S,T,A <: AbstractVector{T}} <: AbstractVector{T}
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

function SplitDataset{S}(chunks::Vector{A}) where {S, T, A <: AbstractVector{T}}
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

function split(
    data::AbstractVector{T},
    splitsize::Integer,
    max_fast_bytes::Integer;
    fast_allocator = stdallocator,
    slow_allocator = stdallocator,
) where {T}
    # Find appropriate split size.
    num_fast_partitions = div(max_fast_bytes, 2 ^ splitsize)
    elements_per_partition = div(2 ^ splitsize, sizeof(T))

    chunks = Vector{typeof(data)}()
    fast_partitions_allocated = 0
    for batch in batched(eachindex(data), elements_per_partition)
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

    # TODO: More elegant log2 business ...
    SplitDataset{splitsize - Int(log2(sizeof(T)))}(chunks)
end
