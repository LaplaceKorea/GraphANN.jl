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
        @inbounds s += Distances.evaluate(Distances.Euclidean(), a[i], b[i])
    end
    return s
end

