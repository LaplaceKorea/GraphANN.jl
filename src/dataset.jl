# Custom Dataset
struct SplitDataset{S, T, A <: AbstractVector{T}} <: AbstractVector{T}
    chunks::Vector{A}
    length::Int
end

Base.size(A::SplitDataset) = (A.length,)

function Base.getindex(A::SplitDataset{S}, i::Int) where {S}
    #@boundscheck checkbounds(A, i)
    chunks = A.chunks
    mask = (one(S) << S) - one(S)

    _A = @inbounds(chunks[((i - 1) & mask) + 1])
    return @inbounds(_A[(i & ~mask) >> (S - one(S))])
    #return @inbounds(chunks[i & mask][(i & ~mask) >> (S - one(S))])
end

Base.IndexStyle(::Type{<:SplitDataset}) = Base.IndexLinear()

# function Base.pointer(A::SplitDataset, i::Integer)
#     split = A.split
#     return i <= split ? pointer(A.region1, i) : pointer(A.region2, i - split)
# end
