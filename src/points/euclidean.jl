# Points using the euclidean distances metric
struct Euclidean{N,T}
    vals::NTuple{N,T}
end

Euclidean{N,T}() where {N,T} = Euclidean(ntuple(_ -> zero(T), N))

zeroas(::Type{T}, ::Type{Euclidean{N,U}}) where {T,N,U} = Euclidean{N,T}()
zeroas(::Type{T}, x::E) where {T, E <: Euclidean} = zeroas(T, E)

# Generic plus
@generated function Base.:+(x::Euclidean{N}, y::Euclidean{N}) where {N}
    syms = [Symbol("z$i") for i in 1:N]
    exprs = [:($(syms[i]) = x[$i] + y[$i]) for i in 1:N]
    return quote
        $(exprs...)
        Euclidean(($(syms...),))
    end
end

@generated function Base.:/(x::Euclidean{N,T}, y::U) where {N, T, U <: Number}
    syms = [Symbol("z$i") for i in 1:N]
    exprs = [:($(syms[i]) = x[$i] / y) for i in 1:N]
    return quote
        $(exprs...)
        Euclidean(($(syms...),))
    end
end

raw(x::Euclidean) = x.vals

Base.length(::Euclidean{N}) where {N} = N
Base.eltype(::Euclidean{N,T}) where {N,T} = T

@inline Base.getindex(x::Euclidean, i) = getindex(x.vals, i)

function distance(a::E, b::E) where {N, T, E <: Euclidean{N,T}}
    s = zero(T)
    # Using @simd really helps out here with agressive loop unrolling.
    @simd for i in 1:N
        @inbounds s += (a[i] - b[i]) ^ 2
    end
    return s
end

function distance(a::A, b::B) where {N, TA, TB, A <: Euclidean{N, TA}, B <: Euclidean{N, TB}}
    T = promote_type(TA, TB)
    s = zero(T)
    @simd for i in 1:N
        _a = @inbounds convert(T, a[i])
        _b = @inbounds convert(T, b[i])
        s += (_a - _b) ^ 2
    end
    return s
end

# Prefetching
function prefetch(A::AbstractVector{Euclidean{N,T}}, i) where {N,T}
    # Need to prefetch the entire vector
    # Compute how many cache lines are needed.
    # Divide the number of bytes by 64 to get cache lines.
    cache_lines = (N * sizeof(T)) >> 6

    ptr = pointer(A, i)
    for i in 1:cache_lines
        prefetch(ptr + 64 * (i-1))
    end
    return nothing
end

# What is the vector size for various sized primitives
#
# Use a full 512-bit cache line for Float32
vecsize(::Type{Euclidean}, ::Type{Float32}) = 16

# For UInt8, we need to expand individual UInt8's to Int16's.
# Thus, size the vectors for UInt8 to be 32 (half the 512-bit AVX size so that when we
# expand them to Int16's, the whole 512-bit cache line is occupied.
vecsize(::Type{Euclidean}, ::Type{UInt8}) = 32

# Overload vecs loading functions
vecs_read_type(::Type{Euclidean{N,T}}) where {N,T} = T

# function vecs_convert(::Type{Euclidean{N,T}}, buf::Vector{T}) where {N,T}
#     @assert mod(N, vecsize(Euclidean, T)) == 0
#     @assert length(buf) == N
#     return first(reinterpret(Euclidean{N,T}, buf))
# end

function addto!(v::Vector{Euclidean{N,T}}, index, buf::AbstractVector{T}) where {N,T}
    length(buf) == N || error("Lenght of buffer is incorrect!")
    v[index] = Euclidean{N,T}(ntuple(i -> buf[i], Val(N)))
    return 1
end

vecs_reshape(::Type{<:Euclidean}, v, dim) = v

#####
##### Specialize for UInt8
#####

# Turn a `Euclidean{N,UInt8}` into a tuple of `Vec{32,UInt8}`.
# Ideally, the generated code for this should be a no-op
@generated function deconstruct(x::Euclidean{N, UInt8}) where {N}
    s = vecsize(Euclidean, UInt8)
    @assert mod(N, s) == 0
    num_tuples = div(N, s)
    exprs = map(1:32:N) do i
        inds = [:(x[$(i + j)]) for j in 0:(s-1)]
        return :(SIMD.Vec{$s,UInt8}(($(inds...),)))
    end
    return :(($(exprs...),))
end

# ASSUMPTION: Assume N is a multiple of `vecsize(Euclidean, UInt8)`
# If this is not the case, then `deconstruct` will fail.
function distance(a::E, b::E) where {N, E <: Euclidean{N, UInt8}}
    return _distance(deconstruct(a), deconstruct(b))
end

function _distance(a::T, b::T) where {N, T <: NTuple{N, SIMD.Vec{32, UInt8}}}
    Base.@_inline_meta
    s = zero(SIMD.Vec{16,Int32})
    for i in 1:N
        x = convert(SIMD.Vec{32, Int16}, a[i])
        y = convert(SIMD.Vec{32, Int16}, b[i])

        z = x - y
        s = vnni_accumulate(s, z, z)
    end
    return sum(s)
end

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

