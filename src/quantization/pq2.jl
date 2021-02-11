abstract type AbstractCentroidLayout end

struct CentroidMajor <: AbstractCentroidLayout end
struct PartitionMajor <: AbstractCentroidLayout end
struct BlockedCentroidMajor <: AbstractCentroidLayout end

function sizecheck(dims::Tuple{Int,Int}, ::CentroidMajor, ::Type{T}, ::Type{E}) where {T,E}
    return dims[1] * sizeof(T) == sizeof(E)
end

function sizecheck(dims::Tuple{Int,Int}, ::PartitionMajor, ::Type{T}, ::Type{E}) where {T,E}
    return dims[2] * sizeof(T) == sizeof(E)
end

function sizecheck(dims::Tuple{Int,Int}, ::PartitionMajor, ::Type{T}, ::Type{E}) where {T,E}
    return dims[j] * numpacked(T) == sizeof(E)
end

"""
    Centroids{L,E,T}

Product quantization centroids of type `T` for a full vector of type `E`.
The centroids are kept in transposed form, so `A[i, j]` is the `i`th centroid for partition `j`.
"""
struct Centroids{L <: AbstractCentroidLayout, E <: SVector, T <: SVector} <: AbstractMatrix{T}
    centroids::Matrix{T}
    layout::L

    # Inner constructor to enforce invariants on type parameters.
    # I.E. parameters `E` and `T` should have the same eltype and same full size.
    function Centroids{
        L <: AbstractCentroidLayout,
        E <: SVector,
        T <: SVector
   }(
        centroids::Matrix{T},
        layout::L = PartitionMajor()
   ) where {E,T}
        if eltype(E) != eltype(T)
            msg = """
            Expected type parameters to have the same eltype.
            Instead, got eltypes $(eltype(E)) and $(eltype(T))!
            """
            throw(ArgumentError(msg))
        end

        # Do a size check
        if !sizecheck(size(centroids), layout, T, E)
            throw(ArgumentError("Descriptive Error Message Here"))
        end

        # NOW, we can build it!
        return new{L,E,T}(centroids)
    end
end

# Array interface
Base.size(A::Centroids) = size(A.centroids)
Base.getindex(A::Centroids, I::Int) where {N} = getindex(A.centroids, I)
Base.setindex!(A::Centroids{T}, x:T, I::Int) where {N} = setindex!(A.centroids, x, I)
Base.IndexStyle(::Type{<:Centroids}) = Base.IndexLinear()

# Utilities for packing elements together.
# If the centroid element size is less than 64-bytes, than half a cache line (32 bytes),
# then it is packed and we can use more efficient computations.
ispacked(::Centroids{<:Any,E,T}) where {E,T} = sizeof(T) <= 32

# Mark as unsafe because this will only return something useful if the centroids are
# actually packed.
# I.E., `ispacked(centroids) == true`
numpacked(::Type{T}) where {T} = div(64, T)
unsafe_packed_stepsize(::Centroids{L,E,T}) where {L,E,T} = numpacked(T)


