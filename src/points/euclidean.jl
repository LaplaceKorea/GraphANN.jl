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

function Base.rand(rng::Random.AbstractRNG, ::Random.SamplerType{Euclidean{N,T}}) where {N,T}
    return Euclidean{N,T}(rand(rng, SVector{N,T}))
end

Base.zero(::E) where {E <: Euclidean} = zero(E)
Base.zero(::Type{Euclidean{N,T}}) where {N,T} = Euclidean{N,T}()
Base.one(::E) where {E <: Euclidean} = one(E)
Base.one(::Type{Euclidean{N,T}}) where {N,T} = Euclidean{N,T}(ntuple(_ -> one(T), Val(N)))

Base.sizeof(::Type{Euclidean{N,T}}) where {N,T} = N * sizeof(T)
Base.sizeof(x::E) where {E <: Euclidean} = sizeof(E)

Base.length(::Euclidean{N}) where {N} = N
Base.length(::Type{<:Euclidean{N}}) where {N} = N
Base.lastindex(x::Euclidean) = length(x)

Base.eltype(::Euclidean{N,T}) where {N,T} = T
Base.eltype(::Type{Euclidean{N,T}}) where {N,T} = T
# Need to define transpose for `Euclidean` to that transpose works on arrays of `Euclidean`.
# This is because `transpose` is recursive.
Base.transpose(x::Euclidean) = x

### Pretty Printing
function Base.show(io::IO, x::Euclidean{N,T}) where {N,T}
    print(io, "Euclidean{$N,$T} <")
    for (i, v) in enumerate(x)
        print(io, v)
        i == length(x) || print(io, ", ")
    end
    print(io, ">")
    return nothing
end
_Base.zeroas(::Type{T}, ::Type{Euclidean{N,U}}) where {T,N,U} = Euclidean{N,T}()
_Base.zeroas(::Type{T}, x::E) where {T, E <: Euclidean} = zeroas(T, E)

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
##### SIMD Promotion and Stuff
#####

# There are sevarl goals to be acomplished by the code below:
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
const SIMDType{N,T} = Union{Euclidean{N,T}, SIMD.Vec{N,T}}

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

# Generic fallback
distance_type(::Type{A}, ::Type{B}) where {A,B} = nothing

# Hijack short ints to allow emission of VNNI instructions.
distance_type(::Type{UInt8}, ::Type{UInt8}) = Int16
distance_type(::Type{UInt8}, ::Type{Int16}) = Int16

accum_type(::Type{T}) where {T <: SIMD.Vec} = T
accum_type(::Type{SIMD.Vec{32, Int16}}) = SIMD.Vec{16,Int32}

cost_type(::Type{T}) where {T <: SIMDType} = eltype(accum_type(distance_type(T)))
cost_type(::AbstractVector{T}) where {T <: SIMDType} = cost_type(T)

simd_type(::A, ::B) where {A <: SIMDType, B <: SIMDType} = simd_type(A, B)
simd_type(::Type{T}) where {T} = distance_select(T, T, T, T)
function simd_type(::Type{<:SIMDType{N,T1}}, ::Type{<:SIMDType{N,T2}}) where {N,T1,T2}
    T = distance_select(distance_type(T1, T2), distance_type(T2, T1), T1, T2)
    return SIMD.Vec{vector_width(T, Val(N)), T}
end

#####
##### Distance Computation
#####

function _Base.distance(a::A, b::B) where {A <: Euclidean, B <: Euclidean}
    T = simd_type(A, B)
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

