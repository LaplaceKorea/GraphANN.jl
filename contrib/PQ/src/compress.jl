# Compression routine for a dataset.

# Computation of `η` - the coefficient for the parallel loss.
# TODO: Check for `t ≥ norm(x)`
function η(t, x::GraphANN.MaybePtr{SVector{N}}, scale) where {N}
    n = GraphANN._Base.norm(x)
    isapprox(t, n) && return typemax(t)
    t > n && return zero(t)

    v = (t / n) ^ 2
    return (v / (1 - v)) * (scale - 1)
end

struct WeightedSVector{N,T}
    p_par::Float32
    p_perp::Float32
    value::SVector{N,T}
end

function slice(
    data::AbstractVector{SVector{N,T}},
    size::Integer,
    partition,
    norm_cutoff;
    allocator = GraphANN.stdallocator,
    scale = N,
) where {N,T}
    return slice(data, Val(size), partition, norm_cutoff; allocator)
end

function slice(
    data::AbstractVector{SVector{N,T}},
    ::Val{size},
    partition,
    norm_cutoff;
    allocator = GraphANN.stdallocator,
) where {N,T,size}
    sliced = allocator(WeightedSVector{size,T}, length(data))
    subvec = SVector{size,T}

    #_norm_cutoff = norm_cutoff * size / N
    for i in eachindex(data)
        ptr = Ptr{subvec}(pointer(data, i))
        v = unsafe_load(ptr, partition)

        ratio = η(norm_cutoff, data[i], N)
        if ratio == typemax(ratio)
            sliced[i] = WeightedSVector(one(Float32), zero(Float32), v)
        elseif iszero(ratio)
            sliced[i] = WeightedSVector(zero(Float32), zero(Float32), v)
        else
            p_par = ratio / (1 + ratio)
            p_perp = 1 / (1 + ratio)
            sliced[i] = WeightedSVector(p_par, p_perp, v)
        end
    end
    return sliced
end

function loss(
    metric::GraphANN.InnerProduct, x̃::SVector{N,Float32}, x::WeightedSVector{N,T}
) where {N,T}
    @unpack p_par, p_perp, value = x
    # Compute the parallel and perpendicular losses.
    norm = GraphANN._Base.norm(value)
    error = value - x̃

    parallel = -GraphANN._Base.evaluate(metric, error, value) * value / norm
    perpendicular = error - parallel

    parallel_loss = p_par * GraphANN._Base.norm(parallel) ^ 2
    perp_loss = p_perp * GraphANN._Base.norm(perpendicular) ^ 2
    return parallel_loss + perp_loss
end

function nearest(point::WeightedSVector, centroids::AbstractVector)
    minind = 0
    minloss = typemax(Float32)
    @unpack p_par, p_perp = point

    # Ignore small vectors
    iszero(p_par) && iszero(p_perp) && return (index = minind, loss = minloss)
    for i in eachindex(centroids)
        thisloss = loss(GraphANN.InnerProduct(), centroids[i], point)
        if thisloss < minloss
            minind = i
            minloss = thisloss
        end
    end
    return (index = minind, loss = minloss)
end

function compress(
    data::AbstractVector{WeightedSVector{N,T}},
    centroids::AbstractVector{SVector{N,Float32}};
    threshold = 0.5,
    maxiters = 200,
) where {N,T}
    assignments = Vector{Int}(undef, length(data))
    losses = Vector{Float32}(undef, length(data))
    count = 0
    old_loss = Float32(Inf)
    while true
        # Find nearest centroids
        #for i in eachindex(data)
        GraphANN.dynamic_thread(eachindex(data), 128) do i
            nt = nearest(data[i], centroids)
            assignments[i] = nt.index
            losses[i] = nt.loss
        end

        # Update
        change = zero(Float32)
        #for i in eachindex(centroids)
        GraphANN.dynamic_thread(eachindex(centroids)) do i
            centroid = centroids[i]
            new_centroid = update(data, assignments, centroid, i)
            if any(isnan, new_centroid)
                new_centroid = rand(data).value
            end
            change += GraphANN._Base.norm(new_centroid - centroid)
            centroids[i] = new_centroid
        end

        count += 1
        loss = sum(Iterators.filter(!isinf, losses))
        loss_change = old_loss - loss
        @show loss_change
        (abs(loss_change) < threshold || count >= maxiters) && return nothing
        old_loss = loss
    end
end

function update(
    data::AbstractVector{WeightedSVector{N,T}}, assignments, centroid, centroid_index
) where {N,T}

    # Accumulator Variables
    accum_perp = zero(Float32)
    accum_loss = zero(SMatrix{N,N,Float32})
    accum_vec = zero(SVector{N,Float32})

    # Accumulation
    count = 0
    for i in eachindex(data, assignments)
        assignments[i] == centroid_index || continue
        count += 1

        # Found a data point assigned to this centroid
        @unpack p_par, p_perp, value = data[i]
        accum_perp += p_perp

        n = GraphANN._Base.norm(value) ^ 2
        accum_loss += ((p_par - p_perp) / n) * (value * transpose(value))
        accum_vec += p_par * value
    end
    iszero(count) && return centroid

    # Updating
    diag = SDiagonal(ntuple(_->one(Float32), Val(N)))
    local result
    try
        result = (accum_perp * diag + accum_loss) \ accum_vec
    catch e
        @show (accum_perp * diag + accum_loss)
        rethrow(e)
    end
    return result
end
