#####
##### Lloyd's Algorithm
#####

struct KMeansRunner{T <: Union{ThreadLocal, NamedTuple}, N, F}
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

function KMeansRunner(
    data::AbstractVector{SVector{N,T}},
    executor::F = dynamic_thread
) where {N, T, F}
    D = costtype(SVector{N,T}, SVector{N,Float32})
    centroids = SVector{N,Float32}[]
    lloyds_local = _lloyds_local(executor, D, Val(N))
    return KMeansRunner(lloyds_local, centroids, executor)
end

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
            sums = all_tls[1].sums
            points_per = all_tls[1].points_per
            for i in 2:length(all_tls)
                sums += all_tls[i].sums
                points_per += all_tls[i].points_per
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

reset(::Neighbor{I,D}) where {I,D} = Neighbor{I,D}()

# Simple scalar case, just update in place.
function vupdate(minimum::Neighbor{I,D}, x, y, i::Integer; metric = Euclidean()) where {I,D}
    dist = evaluate(metric, x, y)
    new = Neighbor{I,D}(i, dist)
    return min(minimum, new)
end

# Points for Hijacking:
# - `reset` (make mutating but return the original collection)
# - `vupdate` (can also make mutating)
# - `slices` (automatic handling of mixed vectors and matrices
function findnearest!(x, Y, minimum; metric = Euclidean())
    minimum = reset(minimum)
    for i in eachindex(Y)
        minimum = vupdate(minimum, x, Y[i], i; metric = metric)
    end
    return minimum
end

