# Points using the euclidean distances metric
struct Euclidean end

#####
##### Eager Conversion
#####

const SIMDType{N,T} = Union{SVector{N,T}, SIMD.Vec{N,T}}

# Ideally, the generated code for this should be a no-op, it's just awkward because
# Julia doesn't really have a "bitcast" function ...
# This also has the benefit of allowing us to do lazy zero padding if necessary.
function _cast_impl(f, N::Integer, S::Integer, ::Type{T}) where {T}
    num_tuples = ceil(Int, N / S)
    exprs = map(1:num_tuples) do i
        inds = map(1:S) do j
            index = (i - 1) * S + j
            return index <= N ? :(x[$index]) : :(zero($T))
        end
        return :($f(($(inds...),)))
    end
    return :(($(exprs...),))
end

@generated function cast(::Type{E}, x::SIMDType{N,T}) where {N, T, S, E <: SIMDType{S,T}}
    _cast_impl(E, N, S, T)
end

"""
    EagerWrap{V <: SIMDType, K,N,T}

Wrapper for tuple of `SIMD.Vec{N,T}` with length `K`.
When indexing, converts elements to type `V`.

# Examples
```julia-repl
julia> using SIMD
julia> x = rand(GraphANN.Euclidean{12,UInt8})
Euclidean{12,UInt8} <27, 171, 144, 15, 162, 14, 55, 5, 31, 143, 174, 211>

julia> y = GraphANN._Points.EagerWrap{GraphANN.Euclidean{4,Float32}}(x);

julia> y[1]
<4 x Float32>[27.0, 171.0, 144.0, 15.0]

julia> y[2]
<4 x Float32>[162.0, 14.0, 55.0, 5.0]
```
"""
struct EagerWrap{V <: SIMDType,K,N,T}
    vectors::NTuple{K, SIMD.Vec{N,T}}
end
EagerWrap{V}(x::NTuple{K, SIMD.Vec{N,T}}) where {V,K,N,T} = EagerWrap{V,K,N,T}(x)

# Implementation note - even though the `cast` function looks like is should spew out
# a bunch of garbage, we're depending on LLVM to convert this into a series of bitcasts
# and ultimately a no-op.
function EagerWrap{SIMD.Vec{N1,T1}}(x::SVector{N2,T2}) where {N1,T1,N2,T2}
    vectors = cast(SIMD.Vec{N1,T2}, x)
    return EagerWrap{SIMD.Vec{N1,T1}}(vectors)
end

Base.@propagate_inbounds function Base.getindex(x::EagerWrap{V}, i) where {V}
    return convert(V, x.vectors[i])
end

# Non-converting indexing.
# Probably don't need this ...
Base.@propagate_inbounds function Base.getindex(x::EagerWrap{SIMD.Vec{N,T},<:Any,N,T}) where {N,T}
    return x.vectors[i]
end

Base.length(::EagerWrap{V,K}) where {V,K} = K

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
function vector_width(::Type{T}, ::Val{N}) where {T, N}
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
function find_distance_type(::Type{A}, ::Type{B}) where {A, B}
    return distance_select(distance_type(A,B), distance_type(B,A), A, B)
end

# Generic fallback
distance_type(::Type{A}, ::Type{B}) where {A,B} = nothing

# Hijack short ints to allow emission of VNNI instructions.
distance_type(::Type{UInt8}, ::Type{UInt8}) = Int16
distance_type(::Type{UInt8}, ::Type{Int16}) = Int16

"""
    accum_type(x)

Return the appropriate `SIMD.Vec` type to hold partial results when computing distances.
"""
accum_type(::Type{T}) where {T <: SIMD.Vec} = T
accum_type(::Type{SIMD.Vec{32, Int16}}) = SIMD.Vec{16,Int32}

"""
    costtype(x)

Return the type yielded by distance computations involving `x`.
"""
_Base.costtype(::Type{T}) where {T <: SIMDType} = eltype(accum_type(simd_type(T)))
function _Base.costtype(::Type{A}, ::Type{B}) where {A <: SIMDType, B <: SIMDType}
    return eltype(accum_type(simd_type(A, B)))
end

"""
    simd_type(vector_type1, vector_type2)

Return the SIMD vector type (`SIMD.Vec`) to perform distance computations for the given types.
The default behavior is to use Julia's normal promotion for the element type of the vector arguments.
However, this may be customized by extending `distance_type`.
"""
function simd_type(::Type{<:SIMDType{N1,T1}}, ::Type{<:SIMDType{N2,T2}}) where {N1,N2,T1,T2}
    T = find_distance_type(T1, T2)
    return SIMD.Vec{vector_width(T, Val(min(N1,N2))), T}
end
simd_type(::A, ::B) where {A <: SIMDType, B <: SIMDType} = simd_type(A, B)
simd_type(::Type{T}) where {T} = simd_type(T, T)

#####
##### Distance Computation
#####

"""
    distance(a::Euclidean, b::Euclidean)

Return the euclidean distance between `a` and `b`.
Return type can be queried by `costtype(a, b)`.
"""
function _Base.evaluate(metric::Euclidean, a::A, b::B) where {A <: SVector, B <: SVector}
    Base.@_inline_meta
    T = simd_type(A, B)
    return evaluate(metric, EagerWrap{T}(a), EagerWrap{T}(b))
end

function _Base.evaluate(::Euclidean, a::EagerWrap{V,K}, b::EagerWrap{V,K}) where {V, K}
    Base.@_inline_meta
    s = zero(accum_type(V))

    for i in 1:K
        z = @inbounds(a[i] - b[i])
        s = square_accum(z, s)
    end
    return _sum(s)
end


# The generic "sum" function in SIMD.jl is actually really slow.
# Here, we define our own generic reducing sum function which actually ends up being much
# faster ...
function _sum(x::SIMD.Vec{N,T}) where {N,T}
    s = zero(T)
    @inbounds @fastmath for i in 1:N
        s += x[i]
    end
    return s
end

# Specialize to use special AVX instructions.
square(x::T) where {T <: SIMD.Vec} = square_accum(x, zero(accum_type(T)))
square_accum(x::SIMD.Vec, y::SIMD.Vec) = Base.muladd(x, x, y)
function square_accum(x::SIMD.Vec{32,Int16}, y::SIMD.Vec{16,Int32})
    return vnni_accumulate(y, x, x)
end

# VNNI accumulation for 32xInt16.
@static if VERSION >= v"1.6.0-beta1"
    function vnni_accumulate(
        x::SIMD.Vec{16,Int32},
        a::SIMD.Vec{32,Int16},
        b::SIMD.Vec{32,Int16},
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
            Tuple{SIMD.LVec{16,Int32}, SIMD.LVec{32,Int16}, SIMD.LVec{32,Int16}},
            x.data, a.data, b.data,
        )

        return SIMD.Vec(x)
    end
else
    function vnni_accumulate(
        x::SIMD.Vec{16,Int32},
        a::SIMD.Vec{32,Int16},
        b::SIMD.Vec{32,Int16},
    )
        Base.@_inline_meta

        # Use LLVM call to directly insert the assembly instruction.
        # Don't worry about the conversion from <32 x i16> to <16 x i32>.
        # For some reason, the signature of the LLVM instrinsic wants <16 x i32>, but it's
        # treated correctly by the hardware ...
        #
        # This may be related to C++ AVX intrinsic datatypes being element type agnostic.
        decl = "declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>) #1"
        s = """
            %a1 = bitcast <32 x i16> %1 to <16 x i32>
            %a2 = bitcast <32 x i16> %2 to <16 x i32>

            %val = tail call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %0, <16 x i32> %a1, <16 x i32> %a2) #3
            ret <16 x i32> %val
            """

        # SIMD.Vec's wrap around SIMD.LVec, which Julia knows how to pass correctly to LLVM
        # as raw LLVM vectors.
        x = Base.llvmcall(
            (decl, s),
            SIMD.LVec{16,Int32},
            Tuple{SIMD.LVec{16,Int32}, SIMD.LVec{32,Int16}, SIMD.LVec{32,Int16}},
            x.data, a.data, b.data,
        )

        return SIMD.Vec(x)
    end
end

