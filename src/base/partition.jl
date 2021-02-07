# Utility for parallel partitioning
struct PartitionUtil{T}
    gt::Vector{Int}
    lt::Vector{Int}
    temp::Vector{T}
end

PartitionUtil{T}() where {T} = PartitionUtil(Int[], Int[], T[])
function Base.resize!(x::PartitionUtil, sz::Int)
    resize!(x.gt, sz)
    resize!(x.lt, sz)
    resize!(x.temp, sz)
    return nothing
end

# If we've done a partition, than the index of the split is the last index of the `lt`
# vector plus one.
getsplit(x::PartitionUtil) = (x.lt[end] + 1)

function partition!(
    by::F,
    x::AbstractVector,
    util::PartitionUtil;
    executor::G = dynamic_thread
) where {F,G}
    resize!(util, length(x))
    @unpack gt, lt, temp = util

    # Initial parallel portion.
    executor(eachindex(x, gt, lt, temp), 1024) do i
        y = x[i]
        temp[i] = y
        lt[i], gt[i] = by(y) ? (1, 0) : (0, 1)
    end

    # Compute prefix sum.
    cumsum!(lt, lt)
    gt[1] += lt[end]
    cumsum!(gt, gt)

    # Write back.
    # If `lt[i] > lt[i-1]`, then we know `by(temp[i]) == true`.
    # Thus, we set `x[lt[i]] = temp[i]`.
    # Otherwise, if `lt[i] == lt[i-1]`, then it must be the case that `by(tempi[i]) == false`
    # and the correct index at which to store `temp[i]` is `gt[i]`.
    executor(eachindex(x, gt, lt, temp), 1024) do i
        a = lt[i]
        b = (i == firstindex(lt)) ? 0 : lt[i-1]
        index = (a > b) ? a : gt[i]
        x[index] = temp[i]
    end
    return first(gt)
end
