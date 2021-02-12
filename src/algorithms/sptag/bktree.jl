# Build a BKTree

# High level steps.
# 1. Maintain a permutation vector that tracks positions in a tree to data ids.
# 2. Initialize the tree with root centroids.
# 3.

doubleview(a, b, c) = view(a, view(b, c))

function build_bktree(
    data::AbstractVector{SVector{N,T}};
    fanout = 8,
    leafsize = 32,
    idtype::Type{I} = UInt32,
) where {N,T,I}
    # This serves as a permutation of the indices into the data vector, allowing us to
    # shuffle data around without touching the original vector.
    permutation = collect(I(1):I(length(data)))
    builder = _Trees.TreeBuilder{I}(length(data))

    # Create the tree roots
    centroids = _Quantization.kmeans(data, fanout)
    nearest = vec(bruteforce_search(centroids, data, 1; idtype = I))
    centroids = [data[i] for i in nearest]

    range = _Trees.initnodes!(builder, nearest)
    swap!(permutation, range, nearest)

    dataview = doubleview(data, permutation, remainder(builder))
    counts = _Quantization.parallel_count(centroids, dataview)
    nearest = _Quantization.parallel_nearest(centroids, dataview)



    return permutation, builder.tree
end

function swap!(v::AbstractVector, dst::AbstractVector, src::AbstractVector)
    if length(src) != length(dst)
        throw(ArgumentError("Source and Destination do not have equal lengths!"))
    end

    for (i, j) in zip(src, dst)
        v[i], v[j] = v[j], v[i]
    end
end
