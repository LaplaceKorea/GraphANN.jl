#####
##### Entry Point
#####

"""
    TPTree{K,I}

Preallocation utility for partitioning a dataset using approximate Trinary-projection trees.

## Type Parameters
* `K` - Number of dimensouns to consider for TPT splits.
* `I` - Integer type for index permutations.

## Fields
* `permutation` - Permutation of the indices of a dataset.
* `leafsize` - Maximum size of the leaf groupings of the dataset.
* `numtrials` - Number of random vectors tried to find the weights that yield maximum
    variance across a partition.
* `numsamples` - Maximum number of points sampled per partition.
* `partitioner` - Pre-allocated utility to help split a partition.
* `samples` - Pre-allocated buffer holding the indices of items samples.
* `scratch` - Misc. pre-allocated buffer.
"""
struct TPTree{K,I <: Unsigned}
    permutation::Vector{I}
    # Partition Parameters
    leafsize::Int64
    numtrials::Int64
    numsamples::Int64
    # Scratch Space
    partitioner::PartitionUtil{I}
    samples::Vector{Int}
    scratch::Vector{Int}
end

function TPTree{K,I}(
    length::Integer;
    leafsize = 500,
    numtrials = 100,
    numsamples = 1000,
) where {K, I <: Unsigned}
    permutation = collect(I(1):I(length))
    partitioner = PartitionUtil{I}()
    samples = Int[]
    scratch = Int[]

    return TPTree{K,I}(
        permutation,
        leafsize,
        numtrials,
        numsamples,
        partitioner,
        samples,
        scratch,
    )
end

Base.@propagate_inbounds function evalsplit(
    y::AbstractVector{T1},
    dims::NTuple{K,<:Integer},
    vals::SVector{K,T2},
) where {K,T1,T2}
    s = zero(promote_type(T1, T2))
    for i in 1:K
        s += vals[i] * y[dims[i]]
    end
    return s
end

viewdata(data, tree) = view(data, tree.permutation)
viewdata(data, tree, range::AbstractUnitRange) = view(data, view(tree.permutation, range))

viewperm(tree) = tree.permutation
viewperm(tree, range::AbstractUnitRange) = view(tree.permutation, range)
num_split_dimensions(::TPTree{K}) where {K} = K

_Base.partition!(data, tree::TPTree, x...; kw...) = partition!(println, data, tree, x...; kw...)
function _Base.partition!(
    f::F,
    data::AbstractVector,
    tree::TPTree{K};
    executor = dynamic_thread,
    init = true
) where {F, K}
    # Unload data structures, parameters, and scratch space
    @unpack permutation = tree
    @unpack leafsize, numtrials, numsamples = tree
    @unpack samples, partitioner = tree

    # Resize the permutation vector if requested.
    if init
        resize!(permutation, length(data))
        permutation .= 1:length(data)
    end

    # Do the processing in an explicit stack to avoid ANY potential issues with our
    # stack space blowing up.
    #
    # Bootstrap by queuing the whole dataset.
    stack = [(1, length(data))]
    while !isempty(stack)
        lo, hi = pop!(stack)

        # If we've reached the leaf size, then simply store this pair as a leaf and return.
        if (hi - lo + 1) <= tree.leafsize
            f(lo:hi)
            continue
        end

        # Sample data points in this range and get the dimension indices that have the
        # highest variance
        range = lo:hi
        dims = getdims!(data, tree, range)

        # Among the largest variance dims, generate a partition that maximizes separation.
        weights, mean = getweights!(data, tree, dims, range)

        # Sort `permutation` into two chunks, those whose projection is less than the best mean,
        # and those whose projection is greater than the best mean.
        # Also, we're getting closure inference bugs on `mean` for the anonymous
        # function we're passing to `_Base.partition!`, so wrap this particular calculation
        # in a `let` block to avoid that.
        mid = let mean = mean
            _Base.partition!(
                x -> (evalsplit(data[x], dims, weights) < mean),
                view(permutation, range),
                partitioner;
                executor,
            )
        end

        # Adjust for the fact that `partition!` will return the offset with respect
        # to the range, while we want the split with respect to the current region we
        # are working on.
        mid = mid + lo

        # If for some reason the split is super bad and all the data points end up on
        # one side, then split the region down the middle.
        if mid <= lo || mid >= hi
            mid = div(lo + hi + 1, 2)
            println("Performing partition fallback!")
        end

        # Recurse on the smaller branch first.
        if mid - lo < hi - mid
            push!(stack, (lo, mid - 1))
            push!(stack, (mid, hi))
        else
            push!(stack, (mid, hi))
            push!(stack, (lo, mid - 1))
        end
    end
    return nothing
end

#####
##### Helper Functions
#####

function getdims!(data::AbstractVector, tree::TPTree, range::AbstractUnitRange)
    @unpack samples, numsamples = tree
    dataview = viewdata(data, tree, range)
    resize!(samples, numsamples)
    Random.rand!(samples, eachindex(dataview))
    # Statistics.var will use an efficient online algorithm that will in fact comput
    # the element wise variance.
    #variances = Statistics.var(view(dataview, samples))
    _, variances = _Base.meanvar(dataview[i] for i in samples; default = zeros(eltype(data)))
    dims = max_variance_dims(tree, variances)
    return dims
end

function max_variance_dims(tree::TPTree{K}, variances) where {K}
    @unpack scratch = tree
    resize!(scratch, length(variances))
    scratch .= 1:length(variances)

    # Manually invoke `sort!` instead of using `partialsortperm!`.
    # This is because `partialsortperm!` returns a view, which we don't use and doesn't
    # reliably get removed by the compiler.
    sort!(
        scratch,
        Base.Sort.PartialQuickSort(1:K),
        Base.Sort.Perm(Base.Sort.ord(isless, identity, true, Base.Forward), variances)
    )
    resize!(scratch, K)
    return ntuple(i -> scratch[i], Val(K))
end

function getweights!(data::AbstractVector, tree::TPTree{K}, dims, range) where {K}
    @unpack numtrials, samples = tree
    dataview = viewdata(data, tree, range)

    # Prepare some of the local scratch spaces.
    bestweights = @SVector zeros(Float32, K)
    bestvariance = zero(Float32)
    bestmean = zero(Float32)

    # Generate a number of random weights vectors along the axes with the largest
    # variance.
    # Track which weight generates the largest across our sample space.
    for _ in 1:numtrials
        weights = LinearAlgebra.normalize(2 .* rand(SVector{K, Float32}) .- 1)
        mean, variance = _Base.meanvar(
            evalsplit(dataview[i], dims, weights) for i in samples;
            default = zero(Float32)
        )

        # Update to track maximum
        if variance > bestvariance
            bestvariance = variance
            bestweights = weights
            bestmean = mean
        end
    end

    # Now that we have the best weights, compute the mean.
    return bestweights, bestmean
end

