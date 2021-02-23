# Build a BKTree

# High level steps.
# 1. Maintain a permutation vector that tracks positions in a tree to data ids.
# 2. Initialize the tree with root centroids.
doubleview(a, b, c) = view(a, view(b, c))
shrink!(a::AbstractVector, amount::Integer) = resize!(a, length(a) - amount)
shrink!(a::AbstractUnitRange{T}, amount::Integer) where {T} = first(a):(last(a) - T(amount))

function build_bktree(
    data::AbstractVector{SVector{N,T}};
    fanout = 8,
    leafsize = 32,
    stacksplit = 10000,
    idtype::Type{I} = UInt32,
) where {N,T,I}
    D = costtype(SVector{N,T})

    # This serves as a permutation of the indices into the data vector, allowing us to
    # shuffle data around without touching the original vector.
    permutation = collect(I(1):I(length(data)))
    builder = _Trees.TreeBuilder{I}(length(data))

    # Step 1 - coarse grained pass.
    # Single threaded outer loop, multi-threaded inner loops.
    stack = coarse_pass!(builder, permutation, data, fanout, stacksplit)

    # Step 2 - fine-grained pass.
    # Multi-threaded outer loop, single-threaded inner loops.
    fine_pass!(builder, permutation, stack, data, fanout, leafsize, stacksplit)
    return _Trees.finish(builder)
end

function coarse_pass!(
    builder::_Trees.TreeBuilder{I},
    permutation::Vector{I},
    data::AbstractVector{SVector{N,T}},
    fanout::Integer,
    stacksplit::Integer,
) where {I,N,T}
    largestack = [(parent = 0, range = 1:length(data))]
    smallstack = Vector{eltype(largestack)}()

    kmeans_runner = KMeansRunner(data, dynamic_thread)
    exhaustive_runner = ExhaustiveRunner(
        I,
        length(data),
        one;
        executor = dynamic_thread,
        costtype = Float32,
    )

    # Inner function - add a coarsely clustered partition to the small stack.
    process_stack!(
        x -> push!(smallstack, x),
        largestack,
        builder,
        data,
        permutation,
        kmeans_runner,
        exhaustive_runner,
        fanout,
        stacksplit,
    )

    return smallstack
end

function fine_pass!(
    builder::_Trees.TreeBuilder{I},
    permutation::AbstractVector{I},
    stack::AbstractVector,
    data::AbstractVector{SVector{N,T}},
    fanout::Integer,
    leafsize::Integer,
    stacksplit::Integer,
) where {I,N,T}
    kmeans_runners = ThreadLocal(KMeansRunner(data, single_thread))
    exhaustive_runners = ExhaustiveRunner(
        I,
        stacksplit,
        one;
        executor = single_thread,
        costtype = Float32,
    ) |> ThreadLocal

    local_stacks = ThreadLocal(Vector{eltype(stack)}())
    # Create local parent - builder pair stacks for each thread to avoid fighting over
    # the single lock for the top level builder.
    #subtree_stacks = ThreadLocal(Vector{Tuple{Int, _Trees.Tree{I}}}())

    # Multi-thread the outer loop.
    dynamic_thread(eachindex(stack)) do i
        # Create a whole sub-tree for this range to be merged at the very end.
        topparent, toprange = stack[i]
        subbuilder = _Trees.TreeBuilder{I}(length(toprange))
        callback = ((parent, range),) -> _Trees.addnodes!(subbuilder, parent, view(permutation, range))

        # As far as the sub-process is concerned, it's working on an entirely new tree,
        # so set the initial parent to "0".
        #
        # This will get corrected when the splice in this sub-tree into the top level tree.
        local_stack = local_stacks[]
        push!(local_stack, (parent = 0, range = toprange))
        process_stack!(
            callback,
            local_stack,
            subbuilder,
            data,
            permutation,
            kmeans_runners[],
            exhaustive_runners[],
            fanout,
            leafsize,
        )

        _Trees.addtree!(builder, topparent, _Trees.partialfinish(subbuilder))
    end
end

function process_stack!(
    f::F,
    stack::AbstractVector{<:NamedTuple{(:parent,:range)}},
    builder::_Trees.TreeBuilder,
    data::AbstractVector{SVector{N,T}},
    permutation::AbstractVector{<:Integer},
    kmeans_runner,
    bruteforce_runner,
    fanout::Integer,
    leafsize::Integer
) where {F,N,T}
    while !isempty(stack)
        @unpack parent, range = pop!(stack)

        # Leaf check.
        # If the range is less than the target leaf size, call the passed function.
        if length(range) < leafsize
            f((; parent, range))
            continue
        end

        dataview = doubleview(data, permutation, range)
        centroids = kmeans(dataview, kmeans_runner, fanout)
        # Map the indices returned by `kmeans` to their closest data points.
        resize!(bruteforce_runner, length(centroids))
        centroid_indices = search!(
            bruteforce_runner,
            centroids,
            dataview;
            meter = nothing,
        )

        # Handle cases where we have repeated indices.
        # Remember that `bruteforce_search` computes its results in Index-0.
        # Thus, we need to increment every entry in the computed nearest data points.
        unique!(centroid_indices)
        centroid_indices .+= one(eltype(centroid_indices))
        for (i, j) in enumerate(centroid_indices)
            centroids[i] = dataview[j]
        end
        resize!(centroids, length(centroid_indices))

        # Add the data point indices to the tree, move selected centroids to the end
        # of the current range to aid in procesing the next level.
        permview = view(permutation, range)
        parent_indices = _Trees.addnodes!(
            builder,
            parent,
            (permview[i] for i in centroid_indices),
        )
        move_to_end!(permview, centroid_indices)

        # Now that we've found our centroids and added them to the tree, we need to
        # sort the remaining elements in this range by the centroid they are assigned to.
        # NOTE: Calling `search!` again will invalidate the `centroid_indices` vector,
        # so don't use it any more.
        remaining_range = shrink!(range, length(centroid_indices))
        dataview = doubleview(data, permutation, remaining_range)
        resize!(bruteforce_runner, length(remaining_range))
        assignments = search!(
            bruteforce_runner,
            dataview,
            centroids;
            meter = nothing,
        )
        assignments .+= one(eltype(assignments))

        # Sort data points by the centroid that they map to.
        permview = view(permutation, remaining_range)
        sort!(Dual(assignments, permview); alg = Base.QuickSort, by = first)

        # Finally, scan through the remaining range and assign child ranges.
        start = 1
        for i in 1:length(centroids)
            stop = findfirstfrom(!isequal(i), assignments, start)
            # If for some reason - no points were assigned to this centroid, just continue.
            start == stop && continue

            # Affine transformation from local indices to global indices.
            subrange = remaining_range[start]:(remaining_range[stop - 1])
            push!(stack, (parent = parent_indices[i], range = subrange))
            start = stop
        end
    end
end


#####
##### Utils
#####

Base.@propagate_inbounds function move_to_end!(v::AbstractVector, itr)
    # `itr` is usually pretty short, so use InsertionSort as the algorithm.
    sort!(itr; rev = true, alg = Base.InsertionSort)
    last = lastindex(v) + 1
    for (i, j) in enumerate(itr)
        swapindex = last - i
        v[swapindex], v[j] = v[j], v[swapindex]
    end
end

function findfirstfrom(predicate::F, A, start) where {F}
    for i in Int(start):Int(lastindex(A))
        predicate(@inbounds(A[i])) && return i
    end
    return lastindex(A) + 1
end

# Utility to sort one vector with respect to another.
struct Dual{TA, TB, A <: AbstractVector{TA}, B <: AbstractVector{TB}} <: AbstractVector{Tuple{TA,TB}}
    a::A
    b::B

    # Inner constructor to ensure that (at least initially), the lengths of the wrapped
    # arrays are the same.
    function Dual(a::A, b::B) where {TA, TB, A <: AbstractArray{TA}, B <: AbstractArray{TB}}
        if length(a) != length(b)
            throw(ArgumentError("Arrays must be the same length!"))
        end
        return new{TA,TB,A,B}(a, b)
    end
end

Base.size(A::Dual) = size(A.a)
Base.getindex(A::Dual, i::Int) = (A.a[i], A.b[i])
Base.setindex!(A::Dual, (a, b), i::Int) = (A.a[i] = a; A.b[i] = b)
Base.IndexStyle(::Type{<:Dual}) = Base.IndexLinear()

