# Points using the euclidean distances metric
struct Euclidean{N,T}
    vals::NTuple{N,T}
end

@inline Base.getindex(x::Euclidean, i) = getindex(x.vals, i)

function distance(a::E, b::E) where {N, T, E <: Euclidean{N,T}}
    s = zero(T)
    @simd for i in 1:N
        @inbounds s += Distances.evaluate(Distances.Euclidean(), a[i], b[i])
    end
    return s
end

