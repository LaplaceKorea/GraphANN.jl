module SplitVectors

export SplitVector

# When working in memory limited regimes - it's helpful to use as much DRAM as possible
# while still allowing PM to be used.
#
# This is a simple vector type that allows part of its data to live in one memory pool and
# part to live in another memory pool.
struct SplitVector{T, A <: AbstractVector{T}, B <: AbstractVector{T}} <: AbstractVector{T}
    lower::A
    upper::B
end

function SplitVector{T}(
    ::UndefInitializer,
    length_a::Integer,
    allocator_a,
    length_b::Integer,
    allocator_b,
) where {T}
    lower = allocator_a(T, length_a)
    upper = allocator_b(T, length_b)
    return SplitVector(lower, upper)
end

# Array Interface
Base.size(x::SplitVector) = (length(x.lower) + length(x.upper),)

function Base.getindex(x::SplitVector, i::Int)
    @boundscheck checkbounds(x, i)
    llower = length(x.lower)
    return i <= length(x.lower) ? x.lower[i] : x.upper[i + 1 - llower]
end

function Base.setindex!(x::SplitVector, v, i::Int)
    @boundscheck checkbounds(x, i)

    llower = length(x.lower)
    if i <= length(x.lower)
        x.lower[i] = v
    else
        x.upper[i + 1 - llower] = v
    end
    return nothing
end

Base.IndexStyle(::Type{<:SplitVector}) = Base.IndexLinear()

end # module
