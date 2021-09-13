# Compression routine for a dataset.

# Computation of `η` - the coefficient for the parallel loss.
# TODO: Check for `t ≥ norm(x)`
function η(t, x::GraphANN.MaybePtr{SVector{N}}) where {N}
    n = GraphANN._Base.norm(x)
    isapprox(t, n) && return typemax(t)
    t > n && return zero(t)
    v = (t / n)^2
    return (v / (1 - v)) * (N - 1)
end

zero!(x) = (x .= zero(eltype(x)))

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
    xbar = zero(MVector{N2,Float32});
    η = one(Float32),
) where {I<:Integer,N1,N2}
    initialize!(xbar, assignments, centroids)

    # Coordinate Descent
    maxiters = 10
    iter = 0
    minloss = typemax(Float32)
    while true
        changed = false
        @inbounds for partition in Base.OneTo(size(centroids, 2))
            minloss = typemax(Float32)
            minindex = 0

            for centroid_index in Base.OneTo(size(centroids, 1))
                setslice!(xbar, centroids[centroid_index, partition], partition)
                loss = ansitropic_loss(x, xbar, η)
                if loss < minloss
                    minloss = loss
                    minindex = centroid_index
                end
            end
            if minindex != assignments[partition]
                changed = true
                assignments[partition] = minindex
            end

            # Revert our inermediate tracking to the minimum index.
            setslice!(xbar, centroids[minindex, partition], partition)
        end
        iter += 1
        (iter > maxiters || (changed == false)) && break
    end
    return minloss
end

Base.@propagate_inbounds function setslice!(
    x::MVector{N1,T}, y::SVector{N2,T}, index
) where {N1,N2,T}
    base = N2 * (index - 1)
    for i in Base.OneTo(N2)
        x[base + i] = y[i]
    end
end

Base.@propagate_inbounds function getslice(::Type{SVector{N,Float32}}, y::SVector, index::Integer) where {N}
    base = N * (index - 1)
    return SVector(ntuple(i -> Float32(y[base + i]), Val(N)))
end

function pass!(
    data::AbstractVector{SVector{N1,T}},
    centroids::AbstractMatrix{SVector{N2,Float32}};
    η = one(Float32),
) where {N1,N2,T}
    # TODO: Validate sizes of data and centroids
    x̄ = GraphANN.ThreadLocal(zero(MVector{N1,Float32}))
    counts = GraphANN.ThreadLocal(zeros(Int, size(centroids)))

    assignments = ones(UInt32, size(centroids, 2), length(data))
    losses = zeros(Float32, length(data))

    meter = ProgressMeter.Progress(length(data), 1)
    iter = GraphANN.batched(eachindex(data), 256)
    GraphANN.dynamic_thread(iter) do range
        local_counts = counts[]
        for i in range
            va = view(assignments, :, i)
            losses[i] = optimize!(va, centroids, data[i], x̄[]; η)
            for i in eachindex(va)
                local_counts[va[i], i] += 1
            end
        end
        ProgressMeter.next!(meter; step = length(range))
    end
    total_counts = sum(GraphANN.getall(counts))

    return assignments, losses, total_counts
end

#####
##### Centroid update
#####

function accum!(C::AbstractMatrix, A::AbstractMatrix, B::Vector{Int})
    for j in axes(A, 2), i in axes(A, 1)
        C[B[i], B[j]] += A[i, j]
    end
end

struct UpdateRunner
    # Left and Right Accumulation
    left_accum::Matrix{Float32}
    right_accum::Vector{Float32}

    # Scratch space
    scratch::Matrix{Float32}
    columns::Vector{Int}
end

function UpdateRunner(
    vector_dim::Integer,
    num_partitions::Integer,
    centroids_per_partition::Integer,
    centroid_dim::Integer,
)
    flatsize = num_partitions * centroids_per_partition * centroid_dim
    left_accum = zeros(Float32, flatsize, flatsize)
    right_accum = zeros(Float32, flatsize)
    scratch = zeros(Float32, vector_dim, vector_dim)
    columns = Int[]
    return UpdateRunner(left_accum, right_accum, scratch, columns)
end

function reset!(runner::UpdateRunner)
    zero!(runner.left_accum)
    zero!(runner.right_accum)
    zero!(runner.scratch)
    return nothing
end

function outerproduct!(C, A::AbstractVector)
    LoopVectorization.@turbo for n in axes(A, 1), m in axes(A, 1)
        C[m, n] = A[m] * A[n]
    end
end

function process!(
    runner::UpdateRunner,
    x::SVector{N},
    assignments::AbstractVector,
    h₌,
    h₊;
    centroids_per_partition = 256,
    centroid_dim = 4,
) where {N}
    # Step 1 - Fill out the set columns for the "B" matrix
    columns = runner.columns
    resize!(columns, centroid_dim * length(assignments))
    columns_index = 1
    for (i, assignment) in enumerate(assignments)
        for j in Base.OneTo(centroid_dim)
            a = (i - 1) * centroids_per_partition * centroid_dim
            b = (assignment - 1) * centroid_dim
            columns[columns_index] = a + b + j
            columns_index += 1
        end
    end

    # Step 2 - Compute the inner matrix for the left accumulation
    scratch, left_accum = runner.scratch, runner.left_accum
    zero!(scratch)
    outerproduct!(scratch, x)
    scalar = (h₌ - h₊) / GraphANN._Base.norm_square(x)
    for i in eachindex(scratch)
        @inbounds scratch[i] *= scalar
    end

    # Add h₌ along the diagonal
    for i in Base.OneTo(size(scratch, 1))
        scratch[i, i] += h₊
    end

    # Step 3 - Perform the operation `Bᵀ * A * B` operation
    accum!(left_accum, scratch, columns)

    # Step 4 - Complete the right accumulation
    right_accum = runner.right_accum
    for i in Base.OneTo(N)
        right_accum[columns[i]] += h₌ * x[i]
    end

    # Step 5 - Profit!
    return nothing
end

function go!(runner, data, assignments; η = one(Float32))
    reset!(runner)
    ProgressMeter.@showprogress 1 for i in eachindex(data)
        process!(runner, data[i], view(assignments, :, i), η, one(Float32))
    end
    return runner
end

#####
##### Squish
#####

# The matrix/vector combination may contain rows and columns of zeros.
# In order to solve the system of linear equations, we need to make the matrix non-singular
# which means we need to remove all these missing entries, solve the system, and then expand
# the solution back to the original size.
#
# Rows/columns of the large matrix will be zero if a particular centroid is not chosen for
# any data point.
#
# If that is the case, then we will also need to repick any unused centroids.
squish(runner::UpdateRunner) = squish(runner.left_accum, runner.right_accum)
function squish(matrix::AbstractMatrix, vector::AbstractVector)
    @assert size(matrix, 1) == size(matrix, 2)
    @assert size(matrix, 2) == length(vector)

    # Find all columns with at least one entry.
    indices = [i for i in Base.OneTo(size(matrix, 2)) if any(!iszero, view(matrix, :, i))]

    # Fast path - all entries are valid.
    # NOTE: Return type here is not type stable.
    if length(indices) == length(vector)
        return (matrix = matrix, vector = vector, indices = (:))
    end

    new_length = length(indices)
    squished_matrix = similar(matrix, eltype(matrix), (new_length, new_length))
    squished_vector = similar(vector, eltype(vector), new_length)

    matrix_view = view(matrix, indices, indices)
    Threads.@threads for i in eachindex(squished_matrix, matrix_view)
        squished_matrix[i] = matrix_view[i]
    end
    squished_vector .= view(vector, indices)
    return (matrix = squished_matrix, vector = squished_vector, indices = indices)
end

function update_centroids!(centroids, nt::NamedTuple)
    return update_centroids!(centroids, nt.matrix, nt.vector, nt.indices)
end

function update_centroids!(
    centroids::AbstractMatrix{SVector{N,T}}, matrix, vector, indices
) where {N,T}
    # Reinterpret "centroids" appropriately.
    centroid_view = view(reinterpret(T, centroids), indices)
    updates = matrix \ vector
    centroid_view .= updates
    return nothing
end

#####
##### Repick unused centroids
#####

function repick!(
    centroids::AbstractMatrix{SVector{N,Float32}},
    data::AbstractVector{SVector{N1,T}},
    counts::AbstractMatrix,
) where {N,N1,T}
    inds = findall(iszero, counts)
    for ind in inds
        # Convert CartesianIndex to a tuple to get the partition number and the centroid's
        # position in this partition.
        partition = ind[2]
        centroids[ind] = getslice(SVector{N,Float32}, rand(data), partition)
    end
    return length(inds)
end

