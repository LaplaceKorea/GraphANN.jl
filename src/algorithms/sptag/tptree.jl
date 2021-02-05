struct TPTSplit{K,T}
    dims::NTuple{K,Int}
    vals::NTuple{K,T}
    split::T
end

Base.isless(y::AbstractVector, x::TPTSplit) = !isless(x, y)
Base.@propagate_inbounds function Base.isless(x::TPTSplit{K,T1}, y::AbstractVector{T2}) where {K,T1,T2}
    @unpack dims, vals, split = x
    s = zero(promote_type(T1, T2))
    for i in 1:K
        s += vals[i] * y[dims[i]]
    end
    return isless(s, split)
end

#####
##### Entry Point
#####

struct TPTree{K, I, T, V <: AbstractVector{T}}
    indices::Vector{I}
    leaves::Vector{UnitRange{I}}
    data::V
    # Partition Parameters
    leafsize::Int64
    numattempts::Int64
    samplerate::Float64
end

function TPTree{K}(
    data::V;
    idtype::Type{I} = UInt32
) where {K, I, T, V <: AbstractVector{T}}
    indices = collect(I(1):I(length(data)))
    leaves = UnitRange{I}[]
    return TPTree{K,I,T,V}(indices, leaves, data, 10, 10, 0.1)
end

num_split_dimensions(::TPTree{K}) where {K} = K
Base.push!(tree::TPTree, range::UnitRange) = push!(tree.leaves, range)
Base.ndims(tree::TPTree{<:Any, <:Any, T}) where {T} = length(T)

function partition!(tree::TPTree, lo::Integer, hi::Integer)
    @unpack data, leafsize, numattempts, samplerate = tree

    # If we've reached the leaf size, then simply store this pair as a leaf and return.
    if hi - lo <= tree.leafsize
        push!(tree, lo:hi)
        return nothing
    end

    # Compute the variance in each dimension of the data in this range.
    # Need to first compute the range.
    range = lo:hi
    means = sum(astype(Float32), view(data, range)) / length(range)
    # TODO: Replace with "literal_pow".
    variances = sum(x -> _Points.square(x - means), view(data, range))

    # Get the top `K` variances.
    indices = collect(1:ndims(tree))
    partialsort!(indices, num_split_dimensions(tree); by = x -> variances[x], rev = true)
    resize!(indices, num_split_dimensions(tree))
    sort!(indices)

    bestweights = zeros(Float32, num_split_dimensions(tree))
    bestmean = zero(Float32)
    projections = zeros(Float32, length(range))
    for i in 1:numattempts
        weight = 2 * rand(Euclidean{Float32, ndims(tree)}) - 1
    end

    # Try several random weights for the key dimensions.
    # Take the result with the best variance.
    return indices, variances
end
