#####
##### Lloyd's Algorithm
#####

"""
Pre-allocated storage for performing [`kmeans`](@ref) clustering.
"""
struct KMeansRunner{T <: MaybeThreadLocal{NamedTuple}, N, F}
    lloyds_local::T
    centroids::Vector{SVector{N,Float32}}
    executor::F
end

# Query if a runner is single-threaded or not.
is_single_thread(::KMeansRunner{<:ThreadLocal}) = false
is_single_thread(::KMeansRunner{<:NamedTuple}) = true

function _lloyds_local(::typeof(dynamic_thread), ::Type{D}, valn::Val{N}) where {D, N}
    return ThreadLocal(_lloyds_local(single_thread, D, valn))
end

function _lloyds_local(::typeof(single_thread), ::Type{D}, ::Val{N}) where {D, N}
    return (
        minimums = Neighbor{UInt64,D}(),
        sums = SVector{N,Float32}[],
        points_per = Int[]
    )
end

"""
    KMeansRunner(data::AbstractVector, [executor], [metric])

Pre-allocate storage for performing [`kmeans`](@ref) clustering on `data`.
Additional information regarding the `executor` and `metric` may be provided.
"""
function KMeansRunner(
    data::AbstractVector{SVector{N,T}},
    executor::F = dynamic_thread,
    metric::M = Euclidean(),
) where {N, T, F, M}
    D = costtype(metric, SVector{N,T}, SVector{N,Float32})
    centroids = SVector{N,Float32}[]
    lloyds_local = _lloyds_local(executor, D, Val(N))
    return KMeansRunner(lloyds_local, centroids, executor)
end

"""
    kmeans(dataset::AbstractVector, runner::KMeansRunner, num_centroids) -> Vector

Perform kmeans clustering on dataset using a pre-allocated [`KMeansRunner`](@ref).
Return a vector containing the centroids.

# Example
```jldoctest
julia> dataset = GraphANN.sample_dataset();

julia> runner = GraphANN.KMeansRunner(dataset, GraphANN.dynamic_thread, GraphANN.Euclidean());

julia> centroids = GraphANN.kmeans(dataset, runner, 8);

julia> typeof(centroids)
Vector{StaticArrays.SVector{128, Float32}} (alias for Array{StaticArrays.SArray{Tuple{128}, Float32, 1, 128}, 1})

julia> length(centroids)
8
```
"""
function kmeans(
    data::AbstractVector{SVector{N,T}},
    runner::KMeansRunner{<:Any,N},
    num_centroids;
    max_lloyds_iterations = 10,
) where {N,T}
    @unpack centroids, lloyds_local, executor = runner
    # Run Lloyd's algorithm.
    resize!(centroids, num_centroids)
    for i in eachindex(centroids)
        @inbounds centroids[i] = rand(data)
    end

    lloyds!(
        centroids,
        data,
        lloyds_local;
        num_iterations = max_lloyds_iterations,
        executor = executor
    )
    return centroids
end

function lloyds!(
    centroids::AbstractVector,
    data::AbstractVector,
    local_storage::Union{ThreadLocal, NamedTuple};
    executor::F = dynamic_thread,
    num_iterations = 1,
    tol = 1E-3,
) where {F}
    num_centroids = length(centroids)
    # Reset local storage
    for t in getall(local_storage)
        resize!(t.sums, num_centroids)
        zero!(t.sums)
        resize!(t.points_per, num_centroids)
        zero!(t.points_per)
    end

    for iter in 1:num_iterations
        executor(eachindex(data), 1024) do i
            # use `getlocal` instead of `getindex` since we may be running this on a
            # single thread.
            threadlocal = _Base.getlocal(local_storage)
            @unpack minimums = threadlocal
            v = data[i]
            neighbor = findnearest!(v, centroids, minimums)

            # Update local tracking data structures.
            # TODO: Move to own function?
            id = getid(neighbor)
            threadlocal.sums[id] += v
            threadlocal.points_per[id] += 1
        end

        # Kind of an ugly hack for now.
        # Rely on constant propagation to save us!
        # First branch - multithreaded case. Reduce into the first thread-local storage.
        if isa(local_storage, ThreadLocal)
            all_tls = getall(local_storage)

            sums = first(all_tls).sums
            points_per = first(all_tls).points_per
            for _i in Iterators.drop(all_tls, 1)
                sums += _i.sums
                points_per += _i.points_per
            end
        # Second branch - single threaded case. No need for reduction, just unpack the
        # NamedTuple.
        elseif isa(local_storage, NamedTuple)
            @unpack sums, points_per = local_storage
        end

        # Update all centroids to be the mean of their assigned points.
        diff = zero(Float32)
        for i in eachindex(sums, points_per, centroids)
            s = sums[i]
            num_points = points_per[i]
            # TODO: Dummy check for now.
            # Need to determine the correct thing to do if a centroid has no assigned
            # points.
            if iszero(num_points)
                # Not the best fallback in the world ...
                newpoint = rand(data)
            else
                newpoint = s / num_points
            end
            @inbounds centroid = centroids[i]
            diff = max(diff, sum(abs, centroid - newpoint) / sum(abs, centroid))
            centroids[i] = newpoint
        end
        diff <= tol && break
    end
end

# At one point in time, I had ambitions of creating an API for this clustering to allow
# for things like efficient product-quantization clustering by hooking into the correct
# API points.
#
# That didn't end up happening (yet), but that is why some of these inner helper methods
# are constructed the way they are ...
reset(::Neighbor{I,D}) where {I,D} = Neighbor{I,D}()

# Simple scalar case, just update in place.
function vupdate(minimum::Neighbor{I,D}, x, y, i::Integer; metric = Euclidean()) where {I,D}
    dist = evaluate(metric, x, y)
    new = Neighbor{I,D}(i, dist)
    return min(minimum, new)
end

# Points for customization:
# - `reset` (can be mutating but must return the original collection)
# - `vupdate` (can also be mutating)
function findnearest!(x, Y, minimum; metric = Euclidean())
    minimum = reset(minimum)
    for i in eachindex(Y)
        minimum = vupdate(minimum, x, Y[i], i; metric = metric)
    end
    return minimum
end

