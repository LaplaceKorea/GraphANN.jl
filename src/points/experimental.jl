#####
##### Lazy Conversion
#####

struct LazyWrap{V <: SIMDType, E <: SIMDType}
    val::E
end

LazyWrap{V}(val::E) where {V,E} = LazyWrap{V,E}(val)

Base.length(x::LazyWrap{V,E}) where {V,E} = div(length(E), length(V))

function Base.getindex(x::LazyWrap{V}, i) where {N,T,V <: SIMDType{N,T}}
    return convert(V, _getindex(x, N * (i - 1)))
end

@generated function _getindex(x::LazyWrap{V}, i) where {N, T, V <: SIMDType{N,T}}
    inds = [:(@inbounds(v[i + $j])) for j in 1:N]
    return quote
        v = _Points.unwrap(x.val)
        V(($(inds...),))
    end
end

#####
##### Lazy Conversion for Arrays
#####

struct LazyArrayWrap{V <: SIMDType, N, T} <: AbstractMatrix{V}
    parent::Vector{Euclidean{N,T}}
end

Base.size(x::LazyArrayWrap{V,N}) where {V,N} = (div(N, length(V)), length(x.parent))
Base.parent(x::LazyArrayWrap) = x.parent

function LazyArrayWrap{V}(x::Vector{Euclidean{N,T}}) where {V <: SIMDType, N, T}
    return LazyArrayWrap{V,N,T}(x)
end

function Base.getindex(x::LazyArrayWrap{V,N,T}, I::Vararg{Int, 2}) where {NV, V <: SIMDType{NV}, N, T}
    @boundscheck checkbounds(x, I...)
    @unpack parent = x
    ptr = Ptr{Euclidean{NV,T}}(pointer(parent, I[2])) + (I[1] - 1) * sizeof(SIMD.Vec{NV,T})
    return convert(V, unsafe_load(ptr))
end

# How do we pack these items.
# Default definition.
packed_type(::Type{<:Euclidean{<:Any,T}}) where {T} = SIMD.Vec{div(64,sizeof(T)),T}

# specializations
packed_type(::Type{<:Euclidean{<:Any,UInt8}}) = SIMD.Vec{32,UInt8}

#####
##### Packed SIMD
#####

# Pack `K` Euclideans into a single `Vec`
struct Packed{K, E <: Euclidean, V <: SIMD.Vec}
    repr::V

    # Inner constructor that only accepts packing if `E` and `V` have the same eltype.
    function Packed{K,E,V}(repr::V) where {K, T, E <: Euclidean{<:Any,T}, V <: SIMD.Vec{<:Any,T}}
        return new{K,E,V}(repr)
    end
end
Packed(e::Euclidean...) = Packed(e)
function Packed(e::NTuple{K,Euclidean{N,T}}) where {K,N,T}
    V = SIMD.Vec{K * N, T}
    return Packed{K, Euclidean{N,T}, V}(V(_merge(e...)))
end

Base.length(::Type{<:Packed{K}}) where {K} = K
Base.length(x::P) where {P <: Packed} = length(P)
simd_type(::Type{<:Packed{<:Any,<:Any,V}}) where {V} = simd_type(V)
Base.transpose(x::Packed) = x

unwrap(x::Packed) = x.repr
function _Base.distance(A::P, B::P) where {K, E, V, P <: Packed{K, E, V}}
    Base.@_inline_meta
    # Figure out the correct promotion type
    promote_type = simd_type(P)
    a = convert(promote_type, unwrap(A))
    b = convert(promote_type, unwrap(B))
    accumulator = square(a - b)
    return squish(Val(K), accumulator)
end

# Convenience setter
set!(A::AbstractArray{<:Packed}, v, I::CartesianIndex) = set!(A, v, Tuple(I))
set!(A::AbstractArray{<:Packed}, v, I::Integer...) = set!(A, v, I)
Base.@propagate_inbounds function set!(
    A::AbstractArray{Packed{K,E,V},N1},
    v::E,
    I::NTuple{N2,Int}
) where {K,E,V,N1,N2}
    @assert N2 == N1 + 1
    # Destructure tuple
    offset = I[1]
    i = LinearIndices(A)[CartesianIndex(Base.tail(I)...)]
    ptr = pointer(A, i) + (offset-1) * sizeof(E)
    unsafe_store!(Ptr{E}(ptr), v)
end

Base.get(A::AbstractArray{<:Packed}, I::CartesianIndex) = get(A, Tuple(I))
Base.get(A::AbstractArray{<:Packed}, I::Integer...) = get(A, I)
Base.@propagate_inbounds function Base.get(
    A::AbstractArray{Packed{K,E,V},N1},
    I::NTuple{N2,Int}
) where {K,E,V,N1,N2}
    @assert N2 == N1 + 1
    # Destructure tuple
    offset = I[1]
    i = LinearIndices(A)[CartesianIndex(Base.tail(I)...)]
    ptr = pointer(A, i) + (offset-1) * sizeof(E)
    return unsafe_load(Ptr{E}(ptr))
end

#####
##### Cache Line Reduction
#####

@generated function squish(::Val{N}, cacheline::SIMD.Vec{S,T}) where {N,S,T}
    # Optimization - if we're just going to sum across the whole vector, then just do
    # that to avoid any kind of expensive reduction.
    # In practice, this is much faster.
    if N == 1
        return quote
            Base.@_inline_meta
            SIMD.Vec(_sum(cacheline))
        end
    end

    # Otherwise, build up a reduction tree.
    step = div(S,N)
    exprs = map(1:step) do i
        tup = valtuple(i, step, S)
        :(SIMD.shufflevector(cacheline, Val($tup)))
    end
    return quote
        Base.@_inline_meta
        reduce(+, ($(exprs...),))
    end
end

valtuple(start, step, stop) = tuple((start - 1):step:(stop - 1)...)

