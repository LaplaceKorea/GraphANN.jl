module Alternatives

# main dep
using GraphANN: GraphANN

# deps
import StaticArrays: SVector

#####
##### "@simd" annotated functions.
#####

struct NaiveEuclidean end
function GraphANN.evaluate(::NaiveEuclidean, a::SVector{N,A}, b::SVector{N,B}) where {N,A,B}
    T = GraphANN.costtype(SVector{N,A}, SVector{N,B})
    s = zero(T)
    @inbounds @simd for i in eachindex(a, b)
        z = convert(T, a[i]) - convert(T, b[i])
        s += z * z
    end
    return s
end

struct UnsafeNaiveEuclideanDynamic
    length::Int
end
@inline function GraphANN.evaluate(
    metric::UnsafeNaiveEuclideanDynamic, a::SVector{N,A}, b::SVector{N,B}
) where {N,A,B}
    T = GraphANN.costtype(SVector{N,A}, SVector{N,B})
    s = zero(T)
    @inbounds @simd for i in Base.OneTo(metric.length)
        z = convert(T, a[i]) - convert(T, b[i])
        s += z * z
    end
    return s
end

#####
##### No "@simd" annotations
#####

struct EuclideanNoAVX end
function GraphANN.evaluate(::EuclideanNoAVX, a::SVector{N,A}, b::SVector{N,B}) where {N,A,B}
    T = GraphANN.costtype(SVector{N,A}, SVector{N,B})
    s = zero(T)
    @inbounds for i in eachindex(a, b)
        z = convert(T, a[i]) - convert(T, b[i])
        s += z * z
    end
    return s
end

struct UnsafeEuclideanNoAVXDynamic
    length::Int
end
function GraphANN.evaluate(
    metric::UnsafeEuclideanNoAVXDynamic, a::SVector{N,A}, b::SVector{N,B}
) where {N,A,B}
    T = GraphANN.costtype(SVector{N,A}, SVector{N,B})
    s = zero(T)
    @inbounds for i in Base.OneTo(metric.length)
        z = convert(T, a[i]) - convert(T, b[i])
        s += z * z
    end
    return s
end

end # module
