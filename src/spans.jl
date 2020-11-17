module Spans

export Span

# By default, Julia pessimizes direct pointer loads because it doesn't assume alignment.
# VectorizationBase implements the `vload` function, which overrides this behavior and
# returns us to fast indexing!
import VectorizationBase

struct Span{T} <: DenseVector{T}
    ptr::Ptr{T}
    length::Int64
end

# Implement Array Interface
Base.pointer(x::Span) = x.ptr
Base.unsafe_convert(::Type{Ptr{T}}, x::Span{T}) where {T} = pointer(x)
Base.size(x::Span) = (x.length,)
Base.sizeof(x::Span) = prod(size(x)) * sizeof(eltype(x))

Base.elsize(x::Span{T}) where {T} = sizeof(T)
Base.elsize(::Type{Span{T}}) where {T} = sizeof(T)

function Base.getindex(x::Span, i::Int)
    @boundscheck checkbounds(x, i)
    return VectorizationBase.vload(pointer(x), sizeof(eltype(x)) * (i-1))
end

function Base.setindex!(x::Span, v, i::Int)
    @boundscheck checkbounds(x, i)
    return VectorizationBase.vstore!(pointer(x), v, sizeof(eltype(x)) * (i-1))
end

Base.IndexStyle(::Type{<:Span}) = Base.IndexLinear()

#####
##### For testing purposes
#####

broadcast_add(a, b) = a .+ b

end
