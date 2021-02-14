#####
##### Entry Points
#####

struct KMeansRunner{N,T,D,NT <: NamedTuple,U,V}
    # intermediate centroids
    initial::Vector{SVector{N,T}}
    refined::Vector{SVector{N,T}}
    final::Vector{SVector{N,Float32}}
    # Cost of each data point
    costs::Vector{D}
    accumulators::NT
    # Thread local storage for intermediate computations.
    choose_tls::ThreadLocal{U}
    lloyds_tls::ThreadLocal{V}
end

function KMeansRunner(
    data::AbstractVector{SVector{N,T}};
    idtype::Type{I} = UInt32,
) where {N,T,I}
    D = costtype(SVector{N,T}, SVector{N, Float32})
    costs = Vector{D}(undef, length(data))
    counts = zeros(Int, 1)

    # TODO: think of a more compasable way of pre-allocating thread local storage.
    choose_tls = initial_centroids_tls(data, I)
    lloyds_tls = ThreadLocal(;
        minimums = Neighbor{I,D}(),
        sums = zeros(SVector{N,Float32}, 1),
        points_per = zeros(Int, 1),
    )

    initial = SVector{N,T}[]
    refined = SVector{N,T}[]
    final = SVector{N,Float32}[]

    accumulators = (
        sums = zeros(SVector{N,Float32}, 1),
        points_per = zeros(Int, 1)
    )

    return KMeansRunner(initial, refined, final, costs, accumulators, choose_tls, lloyds_tls)
end

function kmeans!(
    data::AbstractVector{SVector{N,T}},
    runner::KMeansRunner{N,T},
    num_centroids;
    initial_oversample = 3,
    initial_iterations = 3,
    max_lloyds_iterations = 10,
    executor = dynamic_thread,
) where {N,T}
    # Oversample while choosing initial centroids.
    initial = runner.initial
    @withtimer "choosing" choose_initial_centroids!(
        data,
        initial,
        runner.costs,
        runner.choose_tls,
        initial_oversample * num_centroids;
        num_iterations = initial_iterations,
    )

    # Refine the initial selection.
    weights = parallel_count(initial, data)
    refined = runner.refined
    resize!(refined, num_centroids)
    @withtimer "refining" refine!(refined, initial, weights)

    # Run Lloyd's algorithm.
    final = runner.final
    resize!(final, length(refined))
    final .= refined
    @withtimer "lloyds" lloyds!(
        final,
        data,
        runner.accumulators,
        runner.lloyds_tls;
        num_iterations = max_lloyds_iterations,
        executor = executor
    )
    return final
end

#####
##### Broadcasting things ...
#####

slices(x::AbstractVector) = eachindex(x)
slice(x::AbstractVector, i::Integer) = x[i]

reset(::Neighbor{T,D}) where {T,D} = Neighbor{T,D}()

#####
##### Pick Initial Centroids
#####

initial_centroids_tls(x::AbstractVector, ::Type{I}) where {I} = initial_centroids_tls(x, x, I)

"""
    initial_centroids_tls(data, centroids, idtype::Type)

Allocate appropriate thread local storage for choosing initial centroids during k-means
clustering.
"""
function initial_centroids_tls(data::AbstractVector, centroids::AbstractVector, ::Type{I}) where {I}
    D = costtype(eltype(data),eltype(centroids))
    return ThreadLocal(; minimum = Neighbor{I,D}(), next = Set{I}())
end

# For the first pass, assume `data` and `centroids` are just Vectors.
function choose_initial_centroids!(
    data::AbstractVector{T},
    centroids::AbstractVector{U},
    costs::AbstractVector,
    tls::ThreadLocal,
    num_centroids::Integer;
    num_iterations = 1,
    idtype::Type{I} = UInt32,
) where {T,U,I}
    empty!(centroids)
    push!(centroids, rand(data))
    resize!(costs, length(data))
    adjustment = ceil(Int, num_centroids / num_iterations)
    next = Set{I}()

    for iteration in 1:num_iterations
        # Compute the costs for all data points.
        #@time computecosts!(costs, centroids, data, tls.minimum)
        slicedata(centroids, data, tls.minimum) do i, minimum
            costs[i] = getdistance(minimum)
        end
        total = parallel_sum(costs)
        # Sample next points.
        parallel_sample!(costs, total, tls.next, adjustment)
        # Union updates
        empty!(next)
        for x in getall(tls)
            union!(next, x.next)
        end
        for i in next
            push!(centroids, data[i])
        end
    end
end

# Simple scalar case, just update in place.
function vupdate(minimum::Neighbor{T,D}, x, y, i::Integer; metric = Euclidean()) where {T,D}
    dist = evaluate(metric, x, y)
    new = Neighbor{T,D}(i, dist)
    return min(minimum, new)
end

# Points for Hijacking:
# - `reset` (make mutating but return the original collection)
# - `vupdate` (can also make mutating)
# - `slices` (automatic handling of mixed vectors and matrices
function findnearest!(x, Y, minimum; metric = Euclidean())
    minimum = reset(minimum)
    for i in slices(Y)
        minimum = vupdate(minimum, x, slice(Y, i), i; metric = metric)
    end
    return minimum
end

function refine!(final, centroids, weights)
    numfinal = length(final)
    costs = Vector{Float32}(undef, size(centroids))
    centroids_found = 1
    final[centroids_found] = rand(centroids)

    while centroids_found < numfinal
        for i in slices(centroids)
            costs[i] = minimum(
                x -> weights[i] * evaluate(Euclidean(), centroids[i], x),
                view(final, 1:centroids_found)
            )
        end
        total_cost = sum(costs)

        # The strategy we use is to generate a random number uniformly between 0 and 1.
        # Then, we move through the `costs` vector, accumulating results as we go until
        # we find the first point where the accumulated value is greater than the
        # threshold.
        #
        # This is the point we choose.
        threshold = rand()
        accumulator = zero(threshold)
        index = 0
        while index <= length(centroids) && accumulator < threshold
            index += 1
            accumulator += costs[index] / total_cost
        end

        centroids_found += 1
        final[centroids_found] = centroids[index]
    end
    return final
end

#####
##### Lloyd's Algorithm
#####

function lloyds!(
    centroids::AbstractArray{T},
    data,
    accumulators::NamedTuple,
    tls::ThreadLocal;
    executor = dynamic_thread,
    num_iterations = 1,
    idtype::Type{I} = UInt32
) where {T,I}
    # for now - just populate randomly.
    num_centroids = length(centroids)
    for i in eachindex(centroids)
        centroids[i] = rand(data)
    end

    # reset thread local storage
    for t in getall(tls)
        resize!(t.sums, num_centroids)
        zero!(t.sums)
        resize!(t.points_per, num_centroids)
        zero!(t.points_per)
    end

    @unpack sums, points_per = accumulators
    resize!(sums, num_centroids)
    resize!(points_per, num_centroids)

    for iter in 1:num_iterations
        executor(slices(data), 1024) do i
            threadlocal = tls[]
            @unpack minimums = threadlocal
            v = slice(data, i)
            neighbor = findnearest!(v, centroids, minimums)

            # Update local tracking data structures.
            # TODO: Move to own function?
            id = getid(neighbor)
            threadlocal.sums[id] += v
            threadlocal.points_per[id] += 1
        end

        zero!(sums)
        zero!(points_per)
        for x in getall(tls)
            for i in 1:num_centroids
                sums[i] += x.sums[i]
                points_per[i] += x.points_per[i]
            end
        end

        # Update all centroids to be the mean of their assigned points.
        for i in eachindex(sums, points_per, centroids)
            sum = sums[i]
            num_points = points_per[i]
            # TODO: Dummy check for now.
            # Need to determine the correct thing to do if a centroid has no assigned
            # points.
            if iszero(num_points)
                # Not the best fallback in the world ...
                centroids[i] = rand(data)
            else
                centroids[i] = sum / num_points
            end
        end
    end
end

#####
##### Parallel Algorithms
#####

function slicedata(
    f::F,
    centroids::AbstractVector,
    data::AbstractVector,
    minimums::ThreadLocal;
    metric = Euclidean()
) where {F}
    dynamic_thread(slices(data), 64) do i
        minimum = findnearest!(slice(data, i), centroids, minimums[]; metric = metric)
        f(i, minimum)
    end
end

"""
    parallel_sum(x::AbstractArray) -> Vector

Return the sum of each column of `x`.
"""
function parallel_sum(x::AbstractVector{T}) where {T}
    tls = ThreadLocal(Ref(zero(widen64(T))))
    dynamic_thread(eachindex(x), 64) do i
        localtotal = tls[]
        localtotal[] += x[i]
    end
    return sum(getindex, getall(tls))
end

"""
    parallel_count(centroids, data) -> Array{Int}

Return an array of `size(centroids)` with the counts of how many points in `data` are
closest to each centroid.
"""
function parallel_count(centroids, data)
    D = costtype(eltype(centroids), eltype(data))
    minimums = ThreadLocal(Neighbor{UInt32,D}())
    counts = ThreadLocal(zeros(Int, size(centroids)))

    slicedata(centroids, data, minimums) do i, minimum
        c = counts[]
        c[getid(minimum)] += 1
    end

    return reduce(+, getall(counts))
end

function parallel_sample!(
    costs::AbstractVector,
    totalcost::Number,
    tls::ThreadLocal{<:Set},
    adjustment::Number
)
    foreach(empty!, getall(tls))
    dynamic_thread(slices(costs), 64) do i
        next = tls[]
        val = slice(costs, i)
        sample_probability = adjustment * val / totalcost
        if rand() < sample_probability
            push!(next, i)
        end
    end
end

