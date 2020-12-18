# @generated utilities
_syms(n::Integer) = [Symbol("z$i") for i in 1:n]
_genindex(num_vectors, i) = [:(x[$j][$i]) for j in 1:num_vectors]

# Points using the euclidean distances metric
struct Euclidean{N,T}
    vals::NTuple{N,T}
end

Euclidean{N,T}() where {N,T} = Euclidean(ntuple(_ -> zero(T), N))

_Base.zeroas(::Type{T}, ::Type{Euclidean{N,U}}) where {T,N,U} = Euclidean{N,T}()
_Base.zeroas(::Type{T}, x::E) where {T, E <: Euclidean} = zeroas(T, E)
Base.zero(::E) where {E <: Euclidean} = zero(E)
Base.zero(::Type{Euclidean{N,T}}) where {N,T} = Euclidean{N,T}()

Base.sizeof(::Type{Euclidean{N,T}}) where {N,T} = N * sizeof(T)
Base.sizeof(x::E) where {E <: Euclidean} = sizeof(E)

Base.length(::Euclidean{N}) where {N} = N
Base.length(::Type{<:Euclidean{N}}) where {N} = N
Base.eltype(::Euclidean{N,T}) where {N,T} = T
Base.eltype(::Type{Euclidean{N,T}}) where {N,T} = T

@inline Base.getindex(x::Euclidean, i) = getindex(x.vals, i)

# Use scalar behavior for broadcasting
Base.broadcastable(x::Euclidean) = (x,)

@generated function Base.merge(x::NTuple{K, Euclidean{N,T}}) where {K,N,T}
    final_param = K * N
    exprs = [:(x[$i].vals...) for i in 1:K]
    return :(Euclidean{$final_param,T}(($(exprs...),)))
end

#####
##### Ops
#####

# Strategy - use a generated function to apply an arbitrary function elementwise to a
# collection of Euclidean vectors.
#
# Implementations for ops such as `+` and `-` can then all use the same code path through
# the `_apply` function below.
@generated function _apply(f::F, x::Euclidean{N}...) where {F,N}
    return _apply_impl(N, length(x))
end

function _apply_impl(vector_length, num_vectors)
    syms = _syms(vector_length)
    exprs = [:($(syms[i]) = f($(_genindex(num_vectors, i)...))) for i in 1:vector_length]
    return quote
        $(exprs...)
        Euclidean(($(syms...),))
    end
end

# Unary ops
Base.convert(::Type{T}, x::Euclidean{N,T}) where {N, T} = x
Base.convert(::Type{T}, x::T) where {T <: Euclidean} = x
Base.convert(::Type{Any}, x::Euclidean) = x
Base.convert(::Type{T}, x::Euclidean) where {T} = _apply(i -> convert(T, i), x)

Base.round(::Type{T}, x::Euclidean{N}) where {T, N} = _apply(i -> round(T, i), x)

lossy_convert(::Type{T}, x::Euclidean) where {T} = convert(T, x)
function lossy_convert(::Type{T}, x::Euclidean{N,U}) where {T <: Integer, N, U <: AbstractFloat}
    return round(T, x)
end

# The `sum` here always returns a Float64.
# This is helpful for cases where vectors are composed of things like UInt8 which can
# easily overflow.
function Base.sum(x::Euclidean{N}) where {N}
    s = zero(Float64)
    for i in 1:N
        s += x[i]
    end
    return s
end

# Binary Ops
for op in [:+, :-]
    @eval begin
        Base.$op(x::Euclidean{N}, y::Euclidean{N}) where {N} = _apply($op, x, y)
    end
end

function Base.:/(x::Euclidean{N,T}, y::U) where {N, T, U <: Number}
    return _apply(i -> i / y, x)
end

#####
##### Distance Computation
#####

_cachelines(::Euclidean{N,T}) where {N,T} = (N * sizeof(T)) >> 6

# Turn a `Euclidean{N,T}` into a tuple of `Vec{vecsize(Euclidean, T),T}`.
# Ideally, the generated code for this should be a no-op, it's just awkward because
# Julia doesn't really have a "bitcast" function ...
function _deconstruct_impl(f, N::Integer, S::Integer)
    @assert mod(N, S) == 0
    num_tuples = div(N, S)

    exprs = map(1:S:N) do i
        inds = [:(x[$(i + j)]) for j in 0:(S-1)]
        return :($f(($(inds...),)))
    end
    return :(($(exprs...),))
end

@generated function deconstruct(::Type{Euclidean{S,T}}, x::Euclidean{N,T}) where {S,T,N}
    _deconstruct_impl(Euclidean{S,T}, N, S)
end

@generated function deconstruct(::Type{SIMD.Vec{S,T}}, x::Euclidean{N,T}) where {S,T,N}
    _deconstruct_impl(SIMD.Vec{S,T}, N, S)
end

# Generic fallback for computing distance between to similar-sized Euclidean points with
# a different numeric type.
_promote_type(x...) = promote_type(x...)
_promote_type(x::Type{T}...) where {T <: Integer} = Int64

function _Base.distance(a::A, b::B) where {N, TA, TB, A <: Euclidean{N, TA}, B <: Euclidean{N, TB}}
    T = _promote_type(TA, TB)
    s = zero(T)
    @simd for i in 1:N
        _a = @inbounds convert(T, a[i])
        _b = @inbounds convert(T, b[i])
        s += (_a - _b) ^ 2
    end
    return s
end

# Prefetching
function _Base.prefetch(A::AbstractVector{Euclidean{N,T}}, i, f::F = _Base.prefetch) where {N,T,F}
    # Need to prefetch the entire vector
    # Compute how many cache lines are needed.
    # Divide the number of bytes by 64 to get cache lines.
    cache_lines = (N * sizeof(T)) >> 6

    ptr = pointer(A, i)
    for i in 1:cache_lines
        f(ptr + 64 * (i-1))
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
_IO.vecs_read_type(::Type{Euclidean{N,T}}) where {N,T} = T

function _IO.addto!(v::Vector{Euclidean{N,T}}, index, buf::AbstractVector{T}) where {N,T}
    length(buf) == N || error("Lenght of buffer is incorrect!")
    v[index] = Euclidean{N,T}(ntuple(i -> buf[i], Val(N)))
    return 1
end

_IO.vecs_reshape(::Type{<:Euclidean}, v, dim) = v

#####
##### Specialize for UInt8
#####

# ASSUMPTION: Assume N is a multiple of `vecsize(Euclidean, UInt8)`
# If this is not the case, then `deconstruct` will fail.
const _FAST_EUCLIDEAN_U8 = Union{
    Euclidean{32,UInt8},
    Euclidean{64,UInt8},
    Euclidean{96,UInt8},
    Euclidean{128,UInt8},
}

function _Base.distance(a::E, b::E) where {E <: _FAST_EUCLIDEAN_U8}
    return _distance(
        deconstruct(SIMD.Vec{32,UInt8}, a),
        deconstruct(SIMD.Vec{32,UInt8}, b),
    )
end

function _distance(a::T, b::T) where {N, T <: NTuple{N, SIMD.Vec{32, UInt8}}}
    Base.@_inline_meta
    s = zero(SIMD.Vec{16,Int32})
    for i in 1:N
        # 256-bits -> 512 bits
        x = convert(SIMD.Vec{32, Int16}, a[i])
        y = convert(SIMD.Vec{32, Int16}, b[i])

        # Subtract than squared-reduction-sum
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

