# Utility for parallel partitioning
"""
    PartitionUtil{T}

Scratch space for [`partition!`](@ref _Base.partition!).
"""
struct PartitionUtil{T}
    gt::Vector{Int}
    lt::Vector{Int}
    temp::Vector{T}
end

"""
    PartitionUtil{T}()

Create a new [`PartitionUtil`](@ref) type for partitioning vectors with eltype `T`.
"""
PartitionUtil{T}() where {T} = PartitionUtil(Int[], Int[], T[])
function Base.resize!(x::PartitionUtil, sz::Int)
    resize!(x.gt, sz)
    resize!(x.lt, sz)
    resize!(x.temp, sz)
    return nothing
end

"""
    partition!(f, A::AbstractVector, [partition_util]; executor) -> Int64

Partition `A` in place by the boolean function `f` such that all elements `x` where
`f(x) == true` are moved to the front and all elements `x` where `f(x) == false` are moved
to the back.

Return the index `i` of the first element in the partitioned vector such that
`f(A[i]) == false`.

## Implementation Notes
* Partitioning is stable.
* Accepts an `executor` keyword argument to allow multi-threading.
* Supports pre-allocated scratch space via the [`PartitionUtil`](@ref) type.
"""
function partition!(
    by::F,
    x::AbstractVector{T},
    util::PartitionUtil = PartitionUtil{T}();
    executor::G = dynamic_thread,
) where {F,T,G}
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
    gt[begin] += lt[end]
    cumsum!(gt, gt)

    # Write back.
    # If `lt[i] > lt[i-1]`, then we know `by(temp[i]) == true`.
    # Thus, we set `x[lt[i]] = temp[i]`.
    # Otherwise, if `lt[i] == lt[i-1]`, then it must be the case that `by(temp[i]) == false`
    # and the correct index at which to store `temp[i]` is `gt[i]`.
    executor(eachindex(x, gt, lt, temp), 1024) do i
        a = lt[i]
        b = (i == firstindex(lt)) ? 0 : lt[i - 1]
        index = (a > b) ? a : gt[i]
        x[index] = temp[i]
    end
    index = findfirst(!isequal(last(lt)), gt)
    return index === nothing ? (lastindex(x) + 1) : gt[index]
end
