module _Spans

# Spans are simply a pointer-length pair to serve as a light weight, bounds checked access
# to memory.
#
# Some things to note
#
# 1. Constructing a Span will NOT keeps its parent memory alive (the garbage collector
# has no knowledge of the span).
#
# Therefore, you must ensure that the memory pointed to outlives the Span.
#
# 2. Spans will only work for `isbits` types (types that are stored contiguously).
# This is NOT yet inforced, but if it becomes a problem - this can be enforced by using
# an inner constructor and checking for the `isbits` property.
export Span

# By default, Julia pessimizes direct pointer loads because it doesn't assume alignment.
# VectorizationBase implements the `vload` function, which overrides this behavior and
# returns us to fast indexing!
import VectorizationBase

# Subtype `DenseVector` to take advantages AbstractArray methods specialized for
# contiguous memory.
struct Span{T} <: DenseVector{T}
    ptr::Ptr{T}
    length::Int64
end

Span(ptr::Ptr{T}, length::Integer) where {T} = Span{T}(ptr, convert(Int64, length))

# Implement Array Interface
Base.pointer(x::Span) = x.ptr
Base.unsafe_convert(::Type{Ptr{T}}, x::Span{T}) where {T} = pointer(x)
Base.size(x::Span) = (x.length,)
Base.sizeof(x::Span) = prod(size(x)) * sizeof(eltype(x))

Base.elsize(x::Span{T}) where {T} = sizeof(T)
Base.elsize(::Type{Span{T}}) where {T} = sizeof(T)

function Base.getindex(x::Span, i::Int)
    @boundscheck checkbounds(x, i)
    # NB: VectorizationBase loads and stores are index-0 instead of index-1!
    return VectorizationBase.vload(pointer(x), sizeof(eltype(x)) * (i-1))
end

function Base.getindex(x::Span{<:NTuple}, i::Int)
    @boundscheck checkbounds(x, i)
    return unsafe_load(pointer(x, i))
end

function Base.setindex!(x::Span, v, i::Int)
    @boundscheck checkbounds(x, i)
    # NB: VectorizationBase loads and stores are index-0 instead of index-1!
    return VectorizationBase.vstore!(pointer(x), v, sizeof(eltype(x)) * (i-1))
end

function Base.setindex!(x::Span{<:NTuple}, v, i::Int)
    @boundscheck checkbounds(x, i)
    return unsafe_store!(pointer(x, i), v)
end

Base.IndexStyle(::Type{<:Span}) = Base.IndexLinear()

#####
##### For testing purposes
#####

# To test that pointer loads and stores are working correctly, inspect the native code
# of this function when working with spans.
#
# The result should contain unrolled loops if the loading and storing code is working
# corretly.
#
# It should be noted that this test is currently broken :(
broadcast_add(a, b) = a .+ b

end
