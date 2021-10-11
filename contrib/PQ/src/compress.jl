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

η(t, x::AbstractVector{<:SVector}) = η(Float32, t, x)
function η(::Type{T}, t, x::AbstractVector{<:SVector{N}}) where {T,N}
    vals = zeros(T, length(x))
    #for i in eachindex(vals, x)
    GraphANN.dynamic_thread(eachindex(vals, x), 2048) do i
        vals[i] = η(t, pointer(x, i))
    end
    return vals
end

#####
##### Utils
#####

zero!(x) = (x .= zero(eltype(x)))

function anisotropic_loss_ref(
    x::StaticVector{N,Float32}, xbar::StaticVector{N,T}, η
) where {N,T}
    # Compute the parallel and perpendicular losses.
    norm = GraphANN._Base.norm_square(x)
    error = x - xbar

    error₌ = GraphANN._Base.evaluate(GraphANN.InnerProduct(), error, x) * x / norm
    error₊ = error - error₌
    loss₌ = GraphANN._Base.norm_square(error₌)
    loss₊ = GraphANN._Base.norm_square(error₊)
    return η * loss₌ + loss₊
end

function anisotropic_loss(
    x::StaticVector{N,Float32},
    xbar::StaticVector{N,T},
    η::Float32,
    norm = GraphANN._Base.norm_square(x)
) where {N,T}
    Base.@_inline_meta
    error = x - xbar
    c = GraphANN._Base.evaluate(GraphANN.InnerProduct(), error, x)
    v = c / norm
    loss₌ = v * c

    loss₊ = GraphANN._Base.norm_square((1 - v) * x - xbar)
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

#####
##### Coordinate-Descent Optimization
#####

function optimize!(
    assignments::AbstractVector{I},
    centroids::AbstractMatrix{SVector{N1,Float32}},
    x::SVector{N2},
    xbar = zero(MVector{N2,Float32});
    η = one(Float32),
) where {I<:Integer,N1,N2}
    initialize!(xbar, assignments, centroids)

    # Coordinate Descent
    maxiters = 20
    iter = 0
    minloss = typemax(Float32)
    xnorm = GraphANN._Base.norm_square(x)
    while true
        changed = false
        @inbounds for partition in Base.OneTo(size(centroids, 2))
            minloss = typemax(Float32)
            minindex = 0

            for centroid_index in Base.OneTo(size(centroids, 1))
                setslice!(xbar, centroids[centroid_index, partition], partition)
                loss = anisotropic_loss(x, xbar, η, xnorm)
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

Base.@propagate_inbounds function getslice(
    ::Type{SVector{N,Float32}}, y::SVector, index::Integer
) where {N}
    base = N * (index - 1)
    return SVector(ntuple(i -> Float32(y[base + i]), Val(N)))
end

maybe_getindex(x::AbstractVector, i) = x[i]
maybe_getindex(x, _) = x

function pass(
    data::AbstractVector{SVector{N1,T}},
    centroids::AbstractMatrix{SVector{N2,Float32}};
    η = one(Float32),
) where {N1,N2,T}
    # TODO: Validate sizes of data and centroids
    xbar = GraphANN.ThreadLocal(zero(MVector{N1,Float32}))
    counts = GraphANN.ThreadLocal(zeros(Int, size(centroids)))
    localassignments = GraphANN.ThreadLocal(zeros(UInt32, size(centroids, 2)))

    assignments = ones(UInt32, size(centroids, 2), length(data))
    losses = zeros(Float32, length(data))

    meter = ProgressMeter.Progress(length(data), 1)
    iter = GraphANN.batched(eachindex(data), 1024)
    GraphANN.dynamic_thread(iter) do range
    #for range in iter
        local_counts = counts[]

        for i in range
            # Copy assignments sub-vector into a local buffer to avoid cache
            # confliction between threads.
            va = view(assignments, :, i)
            losses[i] = optimize!(
                va,
                centroids,
                data[i],
                xbar[];
                η = maybe_getindex(η, i),
            )

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

function outerproduct!(C, A::AbstractVector)
    LoopVectorization.@turbo for n in axes(A, 1), m in axes(A, 1)
        C[m, n] = A[m] * A[n]
    end
end

# The `UpdateRunner` is a struct that independantly accumulates partial updates
# for the final PQ update.
#
# Update to the large final matrix are stored as a vector of the "PendingUpdate" below
# which can be periodically committed to the actual matrix in a parallel way.
struct PendingUpdate
    row::Int
    col::Int
    val::Float32
end

struct UpdateRunner
    # Left and Right Accumulation
    updates::Vector{PendingUpdate}
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
    return UpdateRunner(PendingUpdate[], right_accum, scratch, columns)
end

function reset!(runner::UpdateRunner)
    empty!(runner.updates)
    zero!(runner.right_accum)
    zero!(runner.scratch)
    return nothing
end

# Process
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
    scratch = runner.scratch
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
    updates = runner.updates
    @inbounds for j in axes(scratch, 2), i in axes(scratch, 1)
        row = columns[i]
        col = columns[j]
        push!(updates, PendingUpdate(row, col, scratch[i, j]))
    end

    # Step 4 - Complete the right accumulation
    right_accum = runner.right_accum
    for i in Base.OneTo(N)
        right_accum[columns[i]] += h₌ * x[i]
    end

    # Step 5 - Profit!
    return nothing
end

# The coordinator wraps around a collection of `UpdateRunners` (one per thread) and
# controls the merging process.
struct RunnerCoordinator{U}
    runners::GraphANN.ThreadLocal{UpdateRunner, U}
    left_accum::Matrix{Float32}
    right_accum::Vector{Float32}
end

function RunnerCoordinator(
    vector_dim::Integer, num_partitions, centroids_per_partition, centroid_dim
)
    runners = GraphANN.ThreadLocal(
        UpdateRunner(vector_dim, num_partitions, centroids_per_partition, centroid_dim)
    )
    flatsize = num_partitions * centroids_per_partition * centroid_dim
    left_accum = zeros(Float32, flatsize, flatsize)
    right_accum = zeros(Float32, flatsize)
    return RunnerCoordinator(runners, left_accum, right_accum)
end

function reset_runners!(coordinator::RunnerCoordinator)
    return foreach(reset!, GraphANN.getall(coordinator.runners))
end

function reset!(coordinator::RunnerCoordinator)
    reset_runners!(coordinator)
    zero!(coordinator.left_accum)
    zero!(coordinator.right_accum)
    return nothing
end

function process!(
    coordinator::RunnerCoordinator,
    data,
    assignments,
    η = one(Float32),
)
    batchsize = 1000 * Threads.nthreads()
    ProgressMeter.@showprogress 1 for range in GraphANN.batched(eachindex(data), batchsize)
        # Step 1 - let each runner process some items.
        reset_runners!(coordinator)
        runners = coordinator.runners
        @time GraphANN.dynamic_thread(range, 1024) do i
            process!(
                runners[],
                data[i],
                view(assignments, :, i),
                maybe_getindex(η, i),
                one(Float32),
            )
        end

        # Step 2 - update the master data `left_accum` matrix.
        # First - we need to partition ranges of the matrix.
        left_accum = coordinator.left_accum
        update_partition_size = GraphANN.cdiv(size(left_accum, 2), Threads.nthreads())
        all_updates = [runner.updates for runner in GraphANN.getall(runners)]
        @time GraphANN.dynamic_thread(eachindex(all_updates)) do i
            updates = all_updates[i]
            for pending_update in updates
                row, col, val = pending_update.row, pending_update.col, pending_update.val
                ind = size(left_accum, 1) * (col - 1) + row
                atomic_ptr_add!(pointer(left_accum, ind), val)
                #left_accum[row, col] += val
            end
        end
        # GraphANN.dynamic_thread(
        #     GraphANN.batched(axes(left_accum, 2), update_partition_size)
        # ) do subrange
        #     # Only update columns in our specified range.
        #     # This ensures that no threads are tromping on eachother.
        #     for vec in all_updates, pending_update in vec
        #         in(pending_update.col, subrange) || continue
        #         row, col, val = pending_update.row, pending_update.col, pending_update.val
        #         left_accum[row, col] += val
        #     end
        # end

        # Step 3 - update the RHS of the system of equations.
        right_accum = coordinator.right_accum
        @time right_accum .+= sum(runner.right_accum for runner in GraphANN.getall(runners))
        println()
    end
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

#####
##### Top Level Entry Point
#####

function anisotropic_pq(
    dataset::AbstractVector{SVector{N,T}},
    centroid_dim::Int,
    ncentroids::Int;
    magnitude_threshold = 0.2,
    relative_loss_threshold = 0.05,
    maxiters = 10,
    savedir = nothing
) where {N,T}
    if !iszero(mod(N, centroid_dim))
        msg = """
        For now, the centroid dimension `$centroid_dim` must evenly divide the data point
        dimension `$N`
        """
        throw(ArgumentError(msg))
    end

    # Lift the centroid dimension into the type domain for type stability.
    return _anisotropic_pq(
        dataset,
        Val(centroid_dim),
        ncentroids,
        Float32(magnitude_threshold);
        relative_loss_threshold,
        maxiters,
        savedir,
    )
end

function _anisotropic_pq(
    dataset::AbstractVector{SVector{N,T}},
    ::Val{K},
    ncentroids,
    magnitude_threshold;
    relative_loss_threshold,
    maxiters,
    savedir = nothing,
) where {N,T,K}
    @info "Calculating η for each data point"
    eta = η(Float32, magnitude_threshold, dataset)
    runner = RunnerCoordinator(N, div(N, K), ncentroids, K)

    @info "Picking Initial Centroids"
    centroids =
        convert.(
            SVector{K,Float32},
            reshape(reinterpret(SVector{K,T}, rand(dataset, ncentroids)), ncentroids, :),
        )



    if savedir !== nothing && !isdir(savedir)
        mkdir(savedir)
    end
    total_loss = Inf32
    local assignments, losses, total_counts
    for i in Base.OneTo(maxiters)
        @info "Beginning iteration $i"
        assignments, losses, total_counts = pass(dataset, centroids; η = eta)
        if savedir !== nothing
            serialize(joinpath(savedir, "assignments_$i.jls"), assignments)
            serialize(joinpath(savedir, "centroids_$i.jls"), centroids)
        end

        this_loss = sum(losses)
        relative_change = abs(this_loss - total_loss) / this_loss

        @info """
        Clustering Done.
        Total Loss: $(this_loss)
        Change since last iteration: $(relative_change)

        Min Centroid Assigments: $(map(minimum, eachcol(total_counts)))
        Max Centroid Assigments: $(map(maximum, eachcol(total_counts)))

        """

        relative_change <= relative_loss_threshold && break

        @info "Computing Centroid updates"
        process!(runner, dataset, assignments, eta)

        @info "Solving system of Linear Equations"
        update_centroids!(centroids, squish(runner.left_accum, runner.right_accum))
        num_repicked = repick!(centroids, dataset, total_counts)

        @info """
        Centroid update done.
        Had to repick $(num_repicked) centroids.
        """
        total_loss = this_loss
    end
    return (; centroids, assignments, losses, total_counts)
end
