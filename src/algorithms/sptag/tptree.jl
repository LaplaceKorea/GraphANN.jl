#####
##### Entry Point
#####

mutable struct TPTreeRunner{K,I}
    permutation::Vector{I}
    partitioner::PartitionUtil{I}
    scratch::Vector{Int}
end

function TPTreeRunner{K}(permutation::AbstractVector{I}) where {K,I}
    return TPTreeRunner{K,I}(permutation, PartitionUtil{I}(), Int[])
end

## Specialize "threadcopy" because when we're copying the TPTreeRunner for thread local
# storage, we still want to maintain the same reference to the underlying permutation.
function _Base.threadcopy(x::TPTreeRunner{K,I}) where {K,I}
    return TPTreeRunner{K,I}(x.permutation, deepcopy(x.partitioner), deepcopy(x.scratch))
end

viewdata(data, runner::TPTreeRunner, range) = doubleview(data, runner.permutation, range)
num_split_dimensions(::TPTreeRunner{K}) where {K} = K
function getsamples(data, runner::TPTreeRunner, range, numsamples)
    start = first(range)
    stop = min(last(range), start + numsamples - 1)
    return viewdata(data, runner, start:stop)
end

Base.@propagate_inbounds function evalsplit(
    y::AbstractVector{T1}, dims::NTuple{K,<:Integer}, vals::SVector{K,T2}
) where {K,T1,T2}
    s = zero(promote_type(T1, T2))
    for i in 1:K
        s += vals[i] * y[dims[i]]
    end
    return s
end

function _Base.partition!(
    data::AbstractVector,
    permutation::AbstractVector{I},
    dimsplit::Val{K};
    leafsize = 1000,
    numtrials = 500,
    numsamples = 1000,
    single_thread_threshold = 10000,
    init = true,
) where {F,I,K}
    # Resize and initialize the permutation vector if requested.
    if init
        resize!(permutation, length(data))
        permutation .= 1:length(data)
    end
    Random.shuffle!(permutation)

    # Like with the BKTree, use a two pass algorithm to improve parllelism.
    # Unload data structures, parameters, and scratch space.
    coarse_runner = TPTreeRunner{K}(permutation)

    # Do the processing in an explicit stack to avoid ANY potential issues with our
    # stack space blowing up.
    #
    # Bootstrap by queuing the whole dataset.
    stack = [(1, length(data))]
    shortstack = Vector{eltype(stack)}()
    @withtimer "TPTree Coarse Pass" process_stack!(
        (lo, hi) -> push!(shortstack, (lo, hi)),
        stack,
        coarse_runner,
        data,
        single_thread_threshold,
        numtrials,
        numsamples;
        executor = dynamic_thread,
    )

    # Now do the fine-grained pass.
    # N.B.: TPTreeRunner has specialized "deepcopy" to make sure it is distributed
    # correctly for thead local storage.
    leaves = ThreadLocal(Vector{UnitRange{Int}}())
    threadlocal = ThreadLocal(;
        localstack = Vector{eltype(stack)}(), runner = TPTreeRunner{K}(permutation)
    )

    @withtimer "TPTree Fine Pass" dynamic_thread(eachindex(shortstack)) do i
        @unpack localstack, runner = threadlocal[]
        push!(localstack, shortstack[i])
        process_stack!(
            (lo, hi) -> push!(leaves[], lo:hi),
            localstack,
            runner,
            data,
            leafsize,
            numtrials,
            numsamples;
            executor = single_thread,
        )
    end
    return reduce(vcat, getall(leaves))
end

function process_stack!(
    f::F,
    stack::AbstractVector{Tuple{Int,Int}},
    runner::TPTreeRunner{K},
    data::AbstractVector,
    leafsize::Integer,
    numtrials::Integer,
    numsamples::Integer;
    executor = dynamic_thread,
) where {F,K}
    @unpack permutation, partitioner = runner
    while !isempty(stack)
        lo, hi = pop!(stack)

        # If we've reached the leaf size, then simply store this pair as a leaf and return.
        if (hi - lo + 1) <= leafsize
            f(lo, hi)
            continue
        end

        # Sample data points in this range and get the dimension indices that have the
        # highest variance
        range = lo:hi
        dims = getdims!(data, runner, range, numsamples)

        # Among the largest variance dims, generate a partition that maximizes separation.
        weights, mean = getweights!(data, runner, dims, range, numsamples, numtrials)

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

function getdims!(
    data::AbstractVector{SVector{N,T}},
    runner::TPTreeRunner,
    range::AbstractUnitRange,
    numsamples::Integer,
) where {N,T}
    samples = getsamples(data, runner, range, numsamples)
    _, variances = _Base.meanvar(SVector{N,Float32}, samples)
    return max_variance_dims(runner, variances)
end

function max_variance_dims(runner::TPTreeRunner{K}, variances) where {K}
    @unpack scratch = runner
    resize!(scratch, length(variances))
    scratch .= 1:length(variances)

    # Manually invoke `sort!` instead of using `partialsortperm!`.
    # This is because `partialsortperm!` returns a view, which we don't use and doesn't
    # reliably get removed by the compiler.
    sort!(
        scratch,
        Base.Sort.PartialQuickSort(1:K),
        Base.Sort.Perm(Base.Sort.ord(isless, identity, true, Base.Forward), variances),
    )
    return ntuple(i -> scratch[i], Val(K))
end

function getweights!(
    data::AbstractVector{SVector{N,T}},
    runner::TPTreeRunner{K},
    dims,
    range,
    numsamples,
    numtrials,
) where {N,T,K}
    samples = getsamples(data, runner, range, numsamples)

    # Prepare some of the local scratch spaces.
    bestweights = @SVector zeros(Float32, K)
    bestvariance = zero(Float32)
    bestmean = zero(Float32)

    # Generate a number of random weights vectors along the axes with the largest
    # variance.
    # Track which weight generates the largest across our sample space.
    for _ in 1:numtrials
        weights = LinearAlgebra.normalize(2 .* rand(SVector{K,Float32}) .- 1)
        mean,
        variance = _Base.meanvar(
            Float32, evalsplit(sample, dims, weights) for sample in samples;
        )

        # Update to track maximum
        if variance > bestvariance
            bestvariance = variance
            bestweights = weights
            bestmean = mean
            #@show bestvariance bestweights bestmean
        end
    end

    # Now that we have the best weights, compute the mean.
    return bestweights, bestmean
end
