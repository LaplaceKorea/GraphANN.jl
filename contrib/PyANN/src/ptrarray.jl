struct PtrVector{T} <: DenseVector{T}
    ptr::Ptr{T}
    length::Int
end

Base.size(x::PtrVector) = (x.length,)
Base.IndexStyle(::PtrVector) = Base.IndexLinear()

Base.unsafe_convert(::Type{Ptr{T}}, A::PtrVector{T}) where {T} = A.ptr
Base.elsize(::Type{PtrVector{T}}) where {T} = sizeof(T)
Base.@propagate_inbounds function Base.getindex(x::PtrVector, i::Int)
    @boundscheck checkbounds(x, i)
    return unsafe_load(pointer(x), i)
end

Base.@propagate_inbounds function Base.setindex!(x::PtrVector, v, i::Int)
    @boundscheck checkbounds(x, i)
    return unsafe_store!(pointer(x), v, i)
end

