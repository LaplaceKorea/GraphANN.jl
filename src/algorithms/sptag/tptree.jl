struct TPTSplit{K,T}
    dims::NTuple{K,Int}
    vals::SVector{K,T}
end

Base.@propagate_inbounds function eval(
    x::TPTSplit{K,T1},
    y::AbstractVector{T2}
) where {K,T1,T2}
    @unpack dims, vals = x
    s = zero(promote_type(T1, T2))
    for i in 1:K
        s += vals[i] * y[dims[i]]
    end
    return s
end

#####
##### Entry Point
#####

struct TPTree{K,I,N,T}
    indices::Vector{I}
    # leaves::ThreadLocal{Vector{UnitRange{I}}}
    data::Vector{SVector{N,T}}
    # Partition Parameters
    leafsize::Int64
    numattempts::Int64
    numsamples::Int64
    # Scratch Space
    partitioner::PartitionUtil{I}
    samples::Vector{Int}
    scratch::Vector{Int}
    bestweights::Vector{Float32}
    projections::Vector{Float32}
end

function TPTree{K}(
    data::Vector{SVector{N,T}};
    idtype::Type{I} = UInt32,
) where {K, N, T, I}
    indices = collect(I(1):I(length(data)))
    partitioner = PartitionUtil{I}()
    samples = Int[]
    scratch = Int[]
    bestweights = Vector{Float32}(undef, K)
    projections = Float32[]

    return TPTree{K,I,N,T}(
        indices,
        data,
        10000,
        100,
        5000,
        partitioner,
        samples,
        scratch,
        bestweights,
        projections,
    )
end

num_split_dimensions(::TPTree{K}) where {K} = K
Base.ndims(tree::TPTree{<:Any, <:Any, N}) where {N} = N

partition!(tree::TPTree, x...; kw...) = partition!((tree, x) -> println(x), tree, x...; kw...)
function partition!(
    f::F,
    tree::TPTree{K},
    lo::Integer,
    hi::Integer;
    executor = dynamic_thread,
) where {F, K}
    # Unload data structures, parameters, and scratch space
    @unpack data, indices = tree
    @unpack leafsize, numattempts, numsamples = tree
    @unpack samples, bestweights, projections, partitioner = tree

    # Do the processing in an explicit stack to avoid ANY potential issues with our
    # stack space blowing up.
    stack = [(lo, hi)]
    while !isempty(stack)
        lo, hi = pop!(stack)

        # If we've reached the leaf size, then simply store this pair as a leaf and return.
        if hi - lo <= tree.leafsize
            f(tree, lo:hi)
            continue
        end

        # Compute the variance in each dimension of the data in this range.
        # Need to first compute the range.
        range = lo:hi
        dataview = view(data, view(indices, range))

        # Get an initial sample for the dimensional mean and variance for the range.
        resize!(samples, numsamples)
        Random.rand!(samples, eachindex(dataview))
        means = Statistics.mean(x -> map(Float32, dataview[x]), samples)
        variances = sum(x -> (dataview[x] .- means) .^ 2, samples)

        # Get the top `K` variances.
        dims = max_variance_dims(tree, variances)

        # Prepare some of the local scratch spaces.
        resize!(bestweights, K)
        zero!(bestweights)
        resize!(projections, numsamples)
        bestmean = zero(Float32)
        bestvariance = zero(Float32)

        # Generate a number of random weights vectors along the axes with the largest
        # variance.
        # Track which weight generates the largest across our sample space.
        for _ in 1:numattempts
            weights = 2 .* rand(SVector{K, Float32}) .- 1
            weights = weights ./ sum(abs, weights)

            splitter = TPTSplit(dims, weights)
            for (index, sample) in pairs(samples)
                @inbounds projections[index] = eval(splitter, dataview[sample])
            end

            meanval = Statistics.mean(projections)
            variance = sum(x -> (x - meanval) ^ 2, projections)

            # Update to track maximum
            if variance > bestvariance
                bestvariance = variance
                bestmean = meanval
                bestweights .= weights
            end
        end

        # Sort `indices` into two chunks, those whose projection is less than the best mean,
        # and those whose projection is greater than the best mean.
        # Also, we're getting closure inference bugs on `bestmean` for the anonymous
        # function we're passing to `_Base.partition!`, so wrap this particular calculation
        # in a `let` block to avoid that.
        g = TPTSplit(dims, SVector{K}(bestweights))
        mid = let bestmean = bestmean
            _Base.partition!(
                x -> (eval(g, data[x]) < bestmean),
                view(indices, range),
                partitioner;
                executor,
            )
        end

        # Adjust for the fact that `partition!` will return the offset with respect
        # to the range, while we want the split with respect to the current region we
        # are working on.
        mid = mid + lo

        # Recurse on the smaller branch first.
        if mid <= lo || mid >= hi
            mid = div(lo + hi + 1, 2)
        end

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

function max_variance_dims(tree::TPTree{K}, variances) where {K}
    @unpack scratch = tree
    resize!(scratch, ndims(tree))
    scratch .= 1:ndims(tree)
    partialsortperm!(scratch, variances, K; rev = true)
    resize!(scratch, K)
    sort!(scratch)

    return ntuple(i -> scratch[i], Val(K))
end

