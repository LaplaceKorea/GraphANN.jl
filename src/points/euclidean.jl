# Points using the euclidean distances metric
struct Euclidean{N,T}
    vals::NTuple{N,T}
end

Euclidean{N,T}() where {N,T} = Euclidean(ntuple(_ -> zero(T), N))

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

#####
##### Specialize for UInt8
#####

# Strategy:
#
# Need to expand the UInt8 bytes to Int16 numbers.
# Then we can safely to the subtraction and VNNI-based accumulation.
function euclidean(
    a::AbstractVector{SIMD.Vec{64,UInt8}},
    b::AbstractVector{SIMD.Vec{64,UInt8}}
)
    # Do lane-wise accumulation and sum all at the end.
    s = zero(SIMD.Vec{16,Int32})
    for i in eachindex(a)
        a1, a2 = @inbounds expand(a[i])
        b1, b2 = @inbounds expand(b[i])

        c1 = a1 - b1
        c2 = a2 - b2

        s = vnni_accumulate(s, c1, c1)
        s = vnni_accumulate(s, c2, c2)
    end
    return sum(s)
end

@generated function mask(::Val{hi}) where {hi}
    exprs = hi ? (32:63) : (0:31)
    return :(Val(($(exprs...,))))
end

# Turn packed uint8's into two int16 vectors
function expand(x::SIMD.Vec{64,UInt8})
    a = convert(SIMD.Vec{32,Int16}, SIMD.shufflevector(x, mask(Val(false))))
    b = convert(SIMD.Vec{32,Int16}, SIMD.shufflevector(x, mask(Val(true))))
    return (a, b)
end

function vnni_accumulate(
    x::SIMD.Vec{16,Int32},
    a::SIMD.Vec{32,Int16},
    b::SIMD.Vec{32,Int16},
)
    Base.@_inline_meta

    # Use LLVM call to directly insert the assembly instruction.
    decl = "declare <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32>, <16 x i32>, <16 x i32>) #1"
    s = """
        %a1 = bitcast <32 x i16> %1 to <16 x i32>
        %a2 = bitcast <32 x i16> %2 to <16 x i32>

        %val = tail call <16 x i32> @llvm.x86.avx512.vpdpwssd.512(<16 x i32> %0, <16 x i32> %a1, <16 x i32> %a2) #3
        ret <16 x i32> %val
        """

    x = Base.llvmcall(
        (decl, s),
        SIMD.LVec{16,Int32},
        Tuple{SIMD.LVec{16,Int32}, SIMD.LVec{32,Int16}, SIMD.LVec{32,Int16}},
        x.data, a.data, b.data,
    )

    return SIMD.Vec(x)
end

# Reference implementation
function euclidean(A::AbstractVector{UInt8}, B::AbstractVector{UInt8})
    s = zero(Float32)
    for (a,b) in zip(A,B)
        c = Float32(a) - Float32(b)
        s += c^2
    end
    return s
end

