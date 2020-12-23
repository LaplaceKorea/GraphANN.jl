#####
##### SIMDLanes
#####

export SIMDLanes

# The goal of this type is to allow arbitrary access to vectors of points.
struct SIMDLanes{T, V <: SIMD.Vec{<:Any,T}, U} <: AbstractMatrix{V}
    vals::Vector{U}
end

function SIMDLanes(::Type{V}, vals::Vector{U}) where {T,V <: SIMD.Vec{<:Any,T},U}
    @assert iszero(mod(sizeof(U), sizeof(V)))
    return SIMDLanes{T,V,U}(vals)
end

@inline Base.pointer(lanes::SIMDLanes, i::Integer) = pointer(lanes.vals, i)

function Base.size(lanes::SIMDLanes{<:Any,V,U}) where {V,U}
    return (div(sizeof(U), sizeof(V)), length(lanes.vals))
end

function Base.getindex(A::SIMDLanes{T,V}, I::Vararg{Int,2}) where {T,V}
    @boundscheck checkbounds(A, I...)

    # Do the pointer type dance required by SIMD.jl
    ptr = convert(Ptr{T}, (pointer(A, I[2]) + (I[1] - 1) * sizeof(V)))
    return SIMD.vload(V, ptr, nothing, Val(true))
end

function Base.setindex!(A::SIMDLanes{T,V}, x, I::Vararg{Int,2}) where {T,V}
    @boundscheck checkbounds(A, I...)

    # Do the pointer type dance required by SIMD.jl
    ptr = convert(Ptr{T}, pointer(A, I[2]) + (I[1] - 1) * sizeof(V))
    return SIMD.vstore(convert(V, x), ptr, nothing, Val(true))
end

