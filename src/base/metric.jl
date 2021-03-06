# Conversion of SVectors to SIMD sized chunks
const SIMDType{N,T} = Union{SVector{N,T},SIMD.Vec{N,T}}
abstract type AbstractWrap{V<:SIMDType,K} end

simdconvert(::Type{V}, x::V) where {V} = x
simdconvert(::Type{V}, x::SIMD.Vec) where {V<:SIMD.Vec} = convert(V, x)
simdconvert(::Type{SIMD.Vec{N,T1}}, x::NTuple{N,T2}) where {N,T1,T2} = SIMD.Vec{N,T1}(x)
simdconvert(::Type{SIMD.Vec{16,Float32}}, x::NTuple{16,Float16}) = cvt_f16_to_f32(x)

# Ideally, the generated code for this should be a no-op, it's just awkward because
# Julia doesn't really have a "bitcast" function ...
# This also has the benefit of allowing us to do lazy zero padding if necessary,
# so that's cool.
#
# Parameters:
# `f`: Conversion function to run on slices of the original SVector.
# `N`: Dimensionality of the original SVector.
# `S`: Dimensionality of the smaller SVectors that we are creating a tuple of.
# `T`: The element types of the SVectors.
function _cast_impl(f, N::Integer, S::Integer, ::Type{T}) where {T}
    num_tuples = ceil(Int, N / S)
    exprs = map(1:num_tuples) do i
        inds = map(1:S) do j
            index = (i - 1) * S + j
            return index <= N ? :(x[$index]) : :(zero($T))
        end
        return :(simdconvert($f, (($(inds...),))))
    end
    return quote
        Base.@_inline_meta
        return ($(exprs...),)
    end
end

@generated function cast(::Type{SIMD.Vec{N,T}}, x::SVector{N0,T0}) where {N,T,N0,T0}
    return _cast_impl(SIMD.Vec{N,T}, N0, N, T0)
end

"""
    ValueWrap{V <: SIMDType, K,N,T}

Wrapper for tuple of `SIMD.Vec{N,T}` with length `K`.
When indexing, converts elements to type `V`.

# Examples
```julia-repl
julia> using SIMD
julia> x = rand(GraphANN.SVector{12,Int8})
12-element StaticArrays.SVector{12, Int8} with indices SOneTo(12):
  -15
   37
  -25
  -77
    8
 -121
   19
  103
  113
   71
  -70
   57

julia> y = GraphANN._Base.ValueWrap{SIMD.Vec{4,Float32}}(x);

julia> y[1]
<4 x Float32>[-15.0, 37.0, -25.0, -77.0]

julia> y[2]
<4 x Float32>[8.0, -121.0, 19.0, 103.0]
```
"""
struct ValueWrap{N,T,K} <: AbstractWrap{SIMD.Vec{N,T},K}
    vectors::NTuple{K,SIMD.Vec{N,T}}
end

function ValueWrap{N,T}(x::SVector) where {N,T}
    Base.@_inline_meta
    vectors = cast(SIMD.Vec{N,T}, x)
    return ValueWrap(vectors)
end

@inline Base.getindex(x::ValueWrap, i) = @inbounds(x.vectors[i])
Base.length(::ValueWrap{<:Any,<:Any,K}) where {K} = K

#####
##### PtrWrap
#####

"""
    PtrWrap{V <: SIMDType, K, N, T, P} <: AbstractWrap{V, K}

Wrap a pointer to data that we which to interpret as `K` copies of a `SIMD.Vec{N,T}` that
will be converted to `V` upon calling `getindex`.

The last type parameter `P` is the number of elements to load from the last chunk in the
case that the original `SVector` was an odd size. If no masking is required, this parameter
will be `0`.
"""
struct PtrWrap{V<:SIMDType,K,N,T,P} <: AbstractWrap{V,K}
    ptr::Ptr{NTuple{N,T}}
end
unwrap(x::PtrWrap) = x.ptr

function PtrWrap{SIMD.Vec{N1,T1}}(ptr::Ptr{SVector{N2,T2}}) where {N1,T1,N2,T2}
    # Number of segments
    K = cdiv(N2, N1)
    # Number of unpadded bits for the last load
    P = mod(N2, N1)
    return PtrWrap{SIMD.Vec{N1,T1},K,N1,T2,P}(convert(Ptr{NTuple{N1,T2}}, ptr))
end

# Maybe mask the the last load
# This may not need to be `@generated`, but I don't really want to have to rely on
# constant propagation for this.
__mask(::PtrWrap{V,K,N,T,0}) where {V,K,N,T} = SIMD.Vec(ntuple(_ -> true, Val(N)))
@generated function __mask(::PtrWrap{V,K,N,T,P}) where {V,K,N,T,P}
    # Only load the lower entries
    tup = ntuple(i -> i <= P ? true : false, N)
    return :(SIMD.Vec($tup))
end

function Base.getindex(x::PtrWrap{V,K,N,T,P}, i) where {V,K,N,T,P}
    mask = ifelse(i == K, __mask(x), SIMD.Vec{N,Bool}(true))
    vec = SIMD.vload(
        SIMD.Vec{N,T}, Ptr{T}(unwrap(x) + (i - 1) * sizeof(SIMD.Vec{N,T})), mask
    )
    return simdconvert(V, vec)
end

# Special case Float16
function Base.getindex(x::PtrWrap{SIMD.Vec{16,Float32},K,16,Float16,P}, i) where {K,P}
    Base.@_inline_meta
    mask = ifelse(i == K, __mask(x), SIMD.Vec{16,Bool}(true))
    vec = SIMD.vload(
        SIMD.Vec{16,Int16},
        Ptr{Int16}(unwrap(x) + (i - 1) * sizeof(NTuple{16,Float16})),
        mask,
    )
    return cvt_f16_to_f32(vec)
end

Base.length(::PtrWrap{V,K}) where {V,K} = K

#####
##### Wrapping
#####

@inline wrap(::Type{SIMD.Vec{N,T}}, x::SVector) where {N,T} = ValueWrap{N,T}(x)
@inline wrap(::Type{V}, x::Ptr{<:SVector}) where {V} = PtrWrap{V}(x)

#####
##### SIMD Promotion and Stuff
#####

# There are several goals to be acomplished by the code below:
#
# - Select the correct type to perform distance computations in.
#   In other words, provide a promotion mechanism that also allows us to customize promotion
#   for types like `UInt8`.
#
# - Select the correct SIMD vector type to facilitate efficient computation.
#   For long vectors, this will usually be a full 64-byte vector.
#   However, for short vectors (less than 64 bytes), it may be more efficient to use
#   128 or 256 bit vectors.
#
#   The mechanism for this is the `vector_width` function below that relies on Julia's
#   agressive constant propagation.
#
# - Actually perform the distance computation in a manner that allows customization.

# Number of bytes in a 128, 256, and 512 bit vector respectively.
const VECTOR_BYTES = (16, 32, 64)

# Constant propagation to the rescue!
function vector_width(::Type{T}, ::Val{N}) where {T,N}
    V = sizeof(T) * N
    if V >= VECTOR_BYTES[3]
        retval = VECTOR_BYTES[3]
    elseif V >= VECTOR_BYTES[2]
        retval = VECTOR_BYTES[2]
    else
        retval = VECTOR_BYTES[1]
    end
    return div(retval, sizeof(T))
end

# Allow only one direction of `distance_type` to be overloaded.
distance_select(::Nothing, ::Nothing, ::Type{A}, ::Type{B}) where {A,B} = promote_type(A, B)
distance_select(::Type{T}, ::Any, ::Any, ::Any) where {T} = T
distance_select(::Nothing, ::Type{T}, ::Any, ::Any) where {T} = T
function find_distance_type(::Type{A}, ::Type{B}) where {A,B}
    return distance_select(distance_type(A, B), distance_type(B, A), A, B)
end

# Generic fallback
distance_type(::Type{A}, ::Type{B}) where {A,B} = nothing

# Hijack short ints to allow emission of VNNI instructions.
const SMALL_INTS = Union{Int8,UInt8,Int16}
distance_type(::Type{<:SMALL_INTS}, ::Type{<:SMALL_INTS}) = Int16

# Float16
distance_type(::Type{Float16}, ::Type{<:Integer}) = Float32
distance_type(::Type{Float16}, ::Type{Float16}) = Float32
distance_type(::Type{Float16}, ::Type{Float32}) = Float32
distance_type(::Type{Float16}, ::Type{Float64}) = Float64

"""
    accum_type(x)

Return the appropriate `SIMD.Vec` type to hold partial results when computing distances.
"""
accum_type(::Type{T}) where {T<:SIMD.Vec} = T
accum_type(::Type{SIMD.Vec{32,Int16}}) = SIMD.Vec{16,Int32}

"""
    simd_type(vector_type1, vector_type2)

Return the SIMD vector type (`SIMD.Vec`) to perform distance computations for the given types.
The default behavior is to use Julia's normal promotion for the element type of the vector arguments.
However, this may be customized by extending `distance_type`.
"""
function simd_type(::Type{<:SIMDType{N1,T1}}, ::Type{<:SIMDType{N2,T2}}) where {N1,N2,T1,T2}
    T = find_distance_type(T1, T2)
    return SIMD.Vec{vector_width(T, Val(min(N1, N2))),T}
end
simd_type(::A, ::B) where {A<:SIMDType,B<:SIMDType} = simd_type(A, B)
simd_type(::Type{T}) where {T} = simd_type(T, T)

#####
##### Distance Computation
#####

abstract type AbstractMetric end

# scalar broadcasting
Base.broadcastable(x::AbstractMetric) = (x,)
Base.lt(metric::AbstractMetric, x, y) = Base.lt(ordering(metric), x, y)

"""
    Euclidean()

When passed to [`evaluate`](@ref), return the square Euclidean distance between two points.

# Example
```jldoctest
julia> a = ones(GraphANN.SVector{4,Float32})
4-element StaticArrays.SVector{4, Float32} with indices SOneTo(4):
 1.0
 1.0
 1.0
 1.0

julia> b = 2 * a
4-element StaticArrays.SVector{4, Float32} with indices SOneTo(4):
 2.0
 2.0
 2.0
 2.0

julia> GraphANN.evaluate(GraphANN.Euclidean(), a, b)
4.0f0
```
"""
struct Euclidean <: AbstractMetric end

"""
    InnerProduct()

When passed to [`evaluate`](@ref), return the negative inner product between two points.
Note: the result is negated since algorithms in GraphANN try to minimize the result
of the metric.

# Example
```jldoctest
julia> a = ones(GraphANN.SVector{4,Float32})
4-element StaticArrays.SVector{4, Float32} with indices SOneTo(4):
 1.0
 1.0
 1.0
 1.0

julia> b = 2 * a
4-element StaticArrays.SVector{4, Float32} with indices SOneTo(4):
 2.0
 2.0
 2.0
 2.0

julia> GraphANN.evaluate(GraphANN.InnerProduct(), a, b)
-4.0f0
```
"""
struct InnerProduct <: AbstractMetric end
ordering(::InnerProduct) = Base.Reverse

"""
    evaluate(metric::AbstractMetric, a::SVector, b::SVector)

Return the distance between `a` and `b`.
Return type can be queried by `costtype(metric, a, b)`.
"""
function evaluate(
    metric::AbstractMetric, a::MaybePtr{A}, b::MaybePtr{B}
) where {A<:SVector,B<:SVector}
    Base.@_inline_meta
    V = simd_type(A, B)
    return evaluate(metric, wrap(V, a), wrap(V, b))
end

# SIMD Patch
#Base.FastMath.sub_fast(x::SIMD.Vec{N,T}, y::SIMD.Vec{N,T}) where {N,T <: Integer} = x - y
function SIMD.Intrinsics.sub(
    x::SIMD.LVec{N,T}, y::SIMD.LVec{N,T}, ::SIMD.Intrinsics.FastMathFlags
) where {N,T<:Integer}
    return SIMD.Vec(x) - SIMD.Vec(y)
end

function evaluate(::Euclidean, a::AbstractWrap{V,K}, b::AbstractWrap{V,K}) where {V,K}
    Base.@_inline_meta
    s = zero(accum_type(V))
    @inbounds @fastmath for i in Base.OneTo(K)
        z = @inbounds(a[i] - b[i])
        s = muladd(z, z, s)
    end
    return _sum(s)
end

function evaluate(
    metric::InnerProduct, a::AbstractWrap{V,K}, b::AbstractWrap{V,K}
) where {V,K}
    Base.@_inline_meta
    return _evaluate(metric, a, b)
end

function _evaluate(::InnerProduct, a::AbstractWrap{V,K}, b::AbstractWrap{V,K}) where {V,K}
    Base.@_inline_meta
    s = zero(accum_type(V))
    @fastmath for i in Base.OneTo(K)
        s = muladd(@inbounds(a[i]), @inbounds(b[i]), s)
    end
    return _sum(s)
end

# The generic "sum" function in SIMD.jl is actually really slow - probably because it has
# better numeric stability or something?
#
# Here, we define our own generic reducing sum function which actually ends up being much
# faster ...
function _sum(x::SIMD.Vec{N,T}) where {N,T}
    s = zero(T)
    @inbounds @fastmath for i in Base.OneTo(N)
        s += x[i]
    end
    return s
end

# # Specialize to use special AVX instructions.
# square(x::T) where {T<:SIMD.Vec} = square_accum(x, zero(accum_type(T)))
# square_accum(x::SIMD.Vec, y::SIMD.Vec) = muladd(x, x, y)

muladd(x::SIMD.Vec, y::SIMD.Vec, z::SIMD.Vec) = Base.muladd(x, y, z)
function muladd(x::SIMD.Vec{32,Int16}, y::SIMD.Vec{32,Int16}, z::SIMD.Vec{16,Int32})
    return vnni_accumulate(z, x, y)
end

# VNNI accumulation for 32xInt16.
function vnni_accumulate(
    x::SIMD.Vec{16,Int32}, a::SIMD.Vec{32,Int16}, b::SIMD.Vec{32,Int16}
)
    Base.@_inline_meta

    # Use LLVM call to directly insert the assembly instruction.
    # Don't worry about the conversion from <32 x i16> to <16 x i32>.
    # For some reason, the signature of the LLVM instrinsic wants <16 x i32>, but it's
    # treated correctly by the hardware ...
    #
    # This may be related to C++ AVX intrinsic datatypes being element type agnostic.
    s = """
        declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>) #0

        define <16 x i32> @entry(<16 x i32>, <32 x i16>, <32 x i16>) #0 {
        top:
            %a1 = bitcast <32 x i16> %1 to <16 x i32>
            %a2 = bitcast <32 x i16> %2 to <16 x i32>

            %val = tail call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %0, <16 x i32> %a1, <16 x i32> %a2) #3
            ret <16 x i32> %val
        }

        attributes #0 = { alwaysinline }
        """

    # SIMD.Vec's wrap around SIMD.LVec, which Julia knows how to pass correctly to LLVM
    # as raw LLVM vectors.
    x = Base.llvmcall(
        (s, "entry"),
        SIMD.LVec{16,Int32},
        Tuple{SIMD.LVec{16,Int32},SIMD.LVec{32,Int16},SIMD.LVec{32,Int16}},
        x.data,
        a.data,
        b.data,
    )

    return SIMD.Vec(x)
end

@inline cvt_f16_to_f32(x::NTuple{16,Float16}) = cvt_f16_to_f32(reinterpret.(Int16, x))
@inline cvt_f16_to_f32(x::NTuple{16,Int16}) = cvt_f16_to_f32(SIMD.Vec(x))
function cvt_f16_to_f32(x::SIMD.Vec{16,Int16})
    Base.@_inline_meta
    s = """
        declare <16 x float> @llvm.x86.vcvtph2ps.512(<16 x i16>) #0
        define <16 x float> @entry(<16 x i16>) #0 {
            %val = tail call <16 x float> @llvm.x86.vcvtph2ps.512(<16 x i16> %0)
            ret <16 x float> %val
        }

        attributes #0 = { alwaysinline }
    """

    y = Base.llvmcall(
        (s, "entry"), SIMD.LVec{16,Float32}, Tuple{SIMD.LVec{16,Int16}}, x.data
    )
    return SIMD.Vec(y)
end

#####
##### Utilities
#####

function norm(x::MaybePtr{T}) where {T<:SVector}
    Base.@_inline_meta
    V = simd_type(T)
    return norm(wrap(V, x))
end

function norm_square(x::MaybePtr{T}) where {T<:SVector}
    Base.@_inline_meta
    V = simd_type(T)
    return norm_square(wrap(V, x))
end

@inline norm(x::AbstractWrap) = sqrt(norm_square(x))
function norm_square(x::AbstractWrap{V,K}) where {V,K}
    Base.@_inline_meta
    s = zero(accum_type(V))
    for i in Base.OneTo(K)
        a = @inbounds(x[i])
        s = muladd(a, a, s)
    end
    return _sum(s)
end

