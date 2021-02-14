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
    idtype::Type{I} = UInt32
) where {N,T,I}
    D = costtype(SVector{N,T})

    # This serves as a permutation of the indices into the data vector, allowing us to
    # shuffle data around without touching the original vector.
    permutation = collect(I(1):I(length(data)))
    builder = _Trees.TreeBuilder{I}(length(data))

    # Pre-allocate storage for bruteforce search.
    bf_groupsize = 32
    bf_tls = _Base.bruteforce_threadlocal(dynamic_thread, I, Float32, 1, bf_groupsize)
    bf_gt = Vector{I}(undef, length(data))

    # initlaize kmeans data
    kmeans_runner = _Quantization.KMeansRunner(data; idtype = I)

    stack = [(parent = 0, range = 1:length(data))]
    meter = ProgressMeter.Progress(length(data), 1)
    while !isempty(stack)
        @unpack parent, range = pop!(stack)
        if length(range) < leafsize
            ProgressMeter.next!(meter; step = length(range))
            _Trees.addnodes!(builder, parent, view(permutation, range))
            shrink!(permutation, length(range))
            continue
        end

        dataview = doubleview(data, permutation, range)
        @withtimer "kmeans" begin
            if length(range) < 4096
                centroids = _Quantization.kmeans!(dataview, kmeans_runner, fanout; executor = single_thread)
            else
                centroids = _Quantization.kmeans!(dataview, kmeans_runner, fanout; executor = dynamic_thread)
            end
        end

        # Find the data points that are closest to the centroids.
        @withtimer "nearest" begin
            resize!(bf_gt, length(centroids))
            bruteforce_search!(
                bf_gt,
                centroids,
                dataview;
                idtype = I,
                tls = bf_tls,
                meter = nothing,
            )
            unique!(bf_gt)
        end

        # Remember that `bruteforce_search` computes its results in Index-0.
        # Thus, we need to increment every entry in the computed nearest data points.
        bf_gt .+= one(eltype(bf_gt))
        for (i, j) in enumerate(bf_gt)
            centroids[i] = data[j]
        end
        resize!(centroids, length(bf_gt))
        ProgressMeter.next!(meter; step = length(centroids))
        usedrange = _Trees.addnodes!(builder, parent, bf_gt)
        move_to_end!(permutation, bf_gt)
        shrink!(permutation, length(bf_gt))

        # Now that we've found our centroids and added them to the tree, we need to
        # sort the remaining elements in this range by the centroid they are assigned to.
        remaining_range = first(range):(last(range) - length(bf_gt))
        remaining_range = shrink!(range, length(bf_gt))
        dataview = doubleview(data, permutation, remaining_range)
        @withtimer "assignments" begin
            resize!(bf_gt, length(remaining_range))
            assignments = bruteforce_search!(
                bf_gt,
                dataview,
                centroids;
                idtype = I,
                tls = bf_tls,
                meter = nothing,
            )
            assignments .+= one(eltype(assignments))
        end

        # Sort by the centroids that are closest.
        permview = view(permutation, remaining_range)
        @withtimer "sorting" sort!(
            Dual(assignments, permview);
            alg = Base.QuickSort,
            by = first,
        )

        # Finally, scan through the remaining range and assign child ranges.
        start = 1
        for i in 1:length(centroids)
            stop = findfirstfrom(!isequal(i), assignments, start)
            start == stop && continue

            subrange = remaining_range[start]:(remaining_range[stop - 1])
            push!(stack, (parent = usedrange[i], range = subrange))
            start = stop
        end
    end
    return builder
end

function move_to_end!(v::AbstractVector, idx)
    last = lastindex(v) + 1
    for (i, j) in enumerate(idx)
        swapindex = last - i
        v[swapindex], v[j] = v[j], v[swapindex]
    end
end

function findfirstfrom(predicate::F, A, start) where {F}
    for i in Int(start):Int(lastindex(A))
        predicate(A[i]) && return i
    end
    return lastindex(A) + 1
end

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

