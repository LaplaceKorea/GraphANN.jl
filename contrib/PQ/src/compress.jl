# Compression routine for a dataset.

# Computation of `η` - the coefficient for the parallel loss.
# TODO: Check for `t ≥ norm(x)`
function η(t, x::GraphANN.MaybePtr{SVector{N}}, scale) where {N}
    n = GraphANN._Base.norm(x)
    isapprox(t, n) && return typemax(t)
    t > n && return zero(t)
    v = (t / n)^2
    return (v / (1 - v)) * (scale - 1)
end

function ansitropic_loss_ref(
    x::StaticVector{N,Float32}, x̄::StaticVector{N,T}, η
) where {N,T}
    # Compute the parallel and perpendicular losses.
    norm = GraphANN._Base.norm_square(x)
    error = x - x̄

    error₌ = -GraphANN._Base.evaluate(GraphANN.InnerProduct(), error, x) * x / norm
    error₊ = error - error₌
    loss₌ = GraphANN._Base.norm_square(error₌)
    loss₊ = GraphANN._Base.norm_square(error₊)
    return η * loss₌ + loss₊
end

function ansitropic_loss(
    x::StaticVector{N,Float32}, x̄::StaticVector{N,T}, η::Float32
) where {N,T}
    error = x - x̄
    norm = GraphANN._Base.norm_square(x)
    c = -GraphANN._Base.evaluate(GraphANN.InnerProduct(), error, x)
    v = c / norm
    loss₌ = v * c

    loss₊ = GraphANN._Base.norm_square((1 - v) * x - x̄)
    return η * loss₌ + loss₊
end

function initialize!(
    xbar::MVector{N2,Float32},
    assignments::AbstractVector{I},
    centroids::AbstractMatrix{SVector{N1,Float32}},
) where {N1,N2,I<:Integer}
    # Initialize
    for j in eachindex(assignments)
        i = assignments[j]
        setslice!(xbar, centroids[i, j], j)
    end
end

function optimize!(
    assignments::AbstractVector{I},
    centroids::AbstractMatrix{SVector{N1,Float32}},
    x::SVector{N2},
    xbar = zero(MVector{N2,Float32}),
) where {I<:Integer,N1,T,N2}
    initialize!(xbar, assignments, centroids)

    # Coordinate Descent
    maxiters = 10
    iter = 0
    while true
        changed = false
        @inbounds for partition in Base.OneTo(size(centroids, 2))
            minloss = typemax(Float32)
            minindex = 0

            for centroid_index in Base.OneTo(size(centroids, 1))
                setslice!(xbar, centroids[centroid_index, partition], partition)
                loss = ansitropic_loss(x, xbar, one(Float32))
                if loss < minloss
                    minloss = loss
                    minindex = centroid_index
                end
            end
            if minindex != assignments[partition]
                changed = true
                assignments[partition] = minindex
            end
        end
        iter += 1
        (iter > maxiters || (changed == false)) && break
    end
end

Base.@propagate_inbounds function setslice!(
    x::MVector{N1,T}, y::SVector{N2,T}, index
) where {N1,N2,T}
    base = N2 * (index - 1)
    for i in Base.OneTo(N2)
        x[base + i] = y[i]
    end
end

function pass!(
    data::AbstractVector{SVector{N1,T}}, centroids::AbstractMatrix{SVector{N2,Float32}}
) where {N1,N2,T}
    # TODO: Validate sizes of data and centroids
    x̄ = GraphANN.ThreadLocal(zero(MVector{N1,Float32}))
    assignments = ones(UInt32, size(centroids, 2), length(data))

    meter = ProgressMeter.Progress(length(data), 1)
    iter = GraphANN.batched(eachindex(data), 128)
    GraphANN.dynamic_thread(iter) do range
        for i in range
            optimize!(view(assignments, :, i), centroids, data[i], x̄[])
        end
        ProgressMeter.next!(meter; step = length(range))
    end
    return assignments
end

