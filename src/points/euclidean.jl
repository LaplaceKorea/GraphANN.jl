# Points using the euclidean distances metric
struct Euclidean{N,T}
    vals::SVector{N,T}
end

unwrap(x::Euclidean) = x.vals
unwrap(x::SIMD.Vec) = x
Base.Tuple(x::Euclidean) = Tuple(unwrap(x))

Euclidean{N,T}() where {N,T} = Euclidean(@SVector zeros(T, N))
Euclidean{N,T}(vals::NTuple{N,T}) where {N,T} = Euclidean(SVector{N,T}(vals))
Euclidean(vals::NTuple{N,T}) where {N,T} = Euclidean{N,T}(vals)
Euclidean(vec::SIMD.Vec) = Euclidean(Tuple(vec))

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
# Need to define transpose for `Euclidean` to that transpose works on arrays of `Euclidean`.
# This is because `transpose` is recursive.
Base.transpose(x::Euclidean) = x

Base.@propagate_inbounds @inline Base.getindex(x::Euclidean, i) = getindex(x.vals, i)

# Use scalar behavior for broadcasting
Base.broadcastable(x::Euclidean) = (x,)
Base.map(f::F, x::Euclidean...) where {F} = Euclidean(map(f, unwrap.(x)...))
Base.:/(x::Euclidean, y::Number) = Euclidean(unwrap(x) ./ y)

Base.:+(x::Euclidean...) = map(+, x...)
Base.:-(x::Euclidean...) = map(-, x...)

Base.iterate(x::Euclidean, s...) = iterate(unwrap(x), s...)

Base.convert(::Type{SIMD.Vec{N,T}}, x::Euclidean{N,T}) where {N,T} = SIMD.Vec{N,T}(Tuple(x))
function Base.convert(::Type{SIMD.Vec{N,T1}}, x::Euclidean{N,T2}) where {N, T1, T2}
    return convert(SIMD.Vec{N,T1}, convert(SIMD.Vec{N,T2}, x))
end
Base.convert(::Type{Euclidean{N,T}}, x::Euclidean{N,T}) where {N,T} = x
function Base.convert(::Type{Euclidean{N,T}}, x::Euclidean{N,U}) where {N,T,U}
    return map(i -> convert(T, i), x)
end

_merge(x, y...) = (Tuple(x)..., _merge(y...)...)
_merge(x) = Tuple(x)

#####
##### SIMD Promotion and Stuff
#####

const SIMDType{N,T} = Union{Euclidean{N,T}, SIMD.Vec{N,T}}

struct Sentinel{A,B} end
distance_parameters(::Type{A}, ::Type{B}) where {A,B} = Sentinel{A,B}()
distance_parameters(::Type{Float32}, ::Type{UInt8}) = (16, Float32)
distance_parameters(::Type{Float32}, ::Type{Float32}) = (16, Float32)
distance_parameters(::Type{UInt8}, ::Type{UInt8}) = (32, Int16)
distance_parameters(::Type{UInt8}, ::Type{Int16}) = (32, Int16)

accum_type(::Type{T}) where {T <: SIMD.Vec} = T
accum_type(::Type{SIMD.Vec{32, Int16}}) = SIMD.Vec{16,Int32}

distance_select(::Sentinel, x) = x
distance_select(x, ::Sentinel) = x
distance_select(x::T, y::T) where {T} = x

function distance_select(::Sentinel{A,B}, ::Sentinel{B,A}) where {A,B}
    throw(ArgumentError("Define `distance_parameters` for types $A and $B"))
end

distance_type(::A, ::B) where {A <: Euclidean, B <: Euclidean} = distance_type(A, B)
distance_type(::Type{T}) where {T <: SIMDType} = distance_type(T, T)
function distance_type(
    ::Type{<:SIMDType{<:Any,A}},
    ::Type{<:SIMDType{<:Any,B}}
) where {A, B}
    # TODO: Maybe add in the option to select smaller AVX vectors to help speed up
    # short vector computations.
    N, T = distance_select(distance_parameters(A, B), distance_parameters(B, A))
    return SIMD.Vec{N,T}
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
end
Packed(e::Euclidean...) = Packed(e)
function Packed(e::NTuple{K,Euclidean{N,T}}) where {K,N,T}
    V = SIMD.Vec{K * N, T}
    return Packed{K, Euclidean{N,T}, V}(V(_merge(e...)))
end

Base.length(::Type{<:Packed{K}}) where {K} = K
Base.length(x::P) where {P <: Packed} = length(P)
distance_type(::Type{<:Packed{<:Any,<:Any,V}}) where {V} = distance_type(V)
Base.transpose(x::Packed) = x

unwrap(x::Packed) = x.repr
function _Base.distance(A::P, B::P) where {K, E, V, P <: Packed{K, E, V}}
    Base.@_inline_meta
    # Figure out the correct promotion type
    promote_type = distance_type(P)
    a = convert(promote_type, unwrap(A))
    b = convert(promote_type, unwrap(B))
    accumulator = square(a - b)
    return squish(Val(K), accumulator)
end

# Convenience setter
set!(A::AbstractArray{<:Packed}, v, I::CartesianIndex) = set!(A, v, Tuple(I))
set!(A::AbstractArray{<:Packed}, v, I::Integer...) = set!(A, v, I)
function set!(
    A::AbstractArray{Packed{K,E,V},N1},
    v::E,
    I::NTuple{N2,Int}
) where {K,E,V,N1,N2}
    @assert N2 == N1 + 1
    # Destructure tuple
    off, i, j = I
    ptr = pointer(A, size(A, 1) * (j-1) + i) + (off-1) * sizeof(E)
    unsafe_store!(Ptr{E}(ptr), v)
end

Base.get(A::AbstractArray{<:Packed}, I::CartesianIndex) = get(A, Tuple(I))
Base.get(A::AbstractArray{<:Packed}, I::Integer...) = get(A, I)
function Base.get(
    A::AbstractArray{Packed{K,E,V},N1},
    I::NTuple{N2,Int}
) where {K,E,V,N1,N2}
    @assert N2 == N1 + 1
    # Destructure tuple
    off, i, j = I
    ptr = pointer(A, size(A, 1) * (j-1) + i) + (off-1) * sizeof(E)
    return unsafe_load(Ptr{E}(ptr))
end

#####
##### Eager Conversion
#####

struct EagerWrap{V <: SIMDType,K,N,T}
    vectors::NTuple{K,SIMD.Vec{N,T}}
end
EagerWrap{V}(x::NTuple{K,SIMD.Vec{N,T}}) where {V,K,N,T} = EagerWrap{V,K,N,T}(x)

function EagerWrap{SIMD.Vec{N1,T1}}(x::Euclidean{N2,T2}) where {N1,T1,N2,T2}
    # Convert `x` into a collection of appropriately sized vectors
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
##### Lazy Conversion
#####

struct LazyWrap{V <: SIMDType, E <: SIMDType}
    val::E
end

LazyWrap{V}(val::E) where {V,E} = LazyWrap{V,E}(val)

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

    ptr = Ptr{Euclidean{NV,T}}(pointer(parent, I[2])) + (I[1] - 1) * sizeof(SIMD.Vec{N,T})
    return convert(V, unsafe_load(ptr))
end

#####
##### Fancy Bitcast
#####

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

@generated function cast(::Type{Euclidean{S,T}}, x::SIMDType{N,T}) where {S,T,N}
    _cast_impl(Euclidean{S,T}, N, S, T)
end

@generated function cast(::Type{SIMD.Vec{S,T}}, x::SIMDType{N,T}) where {S,T,N}
    _cast_impl(SIMD.Vec{S,T}, N, S, T)
end

#####
##### Distance Computation
#####

function _Base.distance(a::A, b::B) where {A <: Euclidean, B <: Euclidean}
    T = distance_type(A, B)
    return distance(EagerWrap{T}(a), EagerWrap{T}(b))
end

function _Base.distance(a::EagerWrap{V,K}, b::EagerWrap{V,K}) where {V, K}
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
square(x::T) where {T} = square_accum(x, zero(accum_type(T)))
square_accum(x, y) = Base.muladd(x, x, y)
function square_accum(x::SIMD.Vec{32,Int16}, y::SIMD.Vec{16,Int32})
    return vnni_accumulate(y, x, x)
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

#####
##### Cache Line Reduction
#####

@generated function squish(::Val{N}, cacheline::SIMD.Vec{S,T}) where {N,S,T}
    step = div(S,N)
    exprs = map(1:step) do i
        tup = valtuple(i, step, S)
        :(SIMD.shufflevector(cacheline, Val($tup)))
    end
    return quote
        #Base.@_inline_meta
        reduce(+, ($(exprs...),))
    end
end

valtuple(start, step, stop) = tuple((start - 1):step:(stop - 1)...)

#####
##### Specialize for UInt8
#####

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

#####
##### Vector Loading Functions
#####

_IO.vecs_read_type(::Type{Euclidean{N,T}}) where {N,T} = T

function _IO.addto!(v::Vector{Euclidean{N,T}}, index, buf::AbstractVector{T}) where {N,T}
    length(buf) == N || error("Lenght of buffer is incorrect!")
    v[index] = Euclidean{N,T}(ntuple(i -> buf[i], Val(N)))
    return 1
end

_IO.vecs_reshape(::Type{<:Euclidean}, v, dim) = v

