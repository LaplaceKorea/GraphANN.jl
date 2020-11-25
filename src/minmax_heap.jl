# This data structure is copied whole-sale from DataStructures.jl
#
# This is because one one of the internal methods is a lot slower than it needs to be,
# and that's fixed in this implementation.
#
# TODO: Open a PR in DataStructures.jl to upstream this change.

################################################
#
# minmax heap type and constructors
#
################################################

struct BinaryMinMaxHeap{T} <: DataStructures.AbstractMinMaxHeap{T}
    valtree::Vector{T}

    BinaryMinMaxHeap{T}() where {T} = new{T}(Vector{T}())

    function BinaryMinMaxHeap{T}(xs::AbstractVector{T}) where {T}
        valtree = _make_binary_minmax_heap(xs)
        new{T}(valtree)
    end
end

BinaryMinMaxHeap(xs::AbstractVector{T}) where T = BinaryMinMaxHeap{T}(xs)

# This is an added method that sorts the heap valtree in place.
# The state of the heap is destroyed, so this is only meant to be called after all items
# have been added and we are just interested in getting the final sorted results.
function destructive_extract!(heap::BinaryMinMaxHeap)
    sort!(heap.valtree; alg = Base.QuickSort)
    return heap.valtree
end

################################################
#
# core implementation
#
################################################

Base.@propagate_inbounds function _make_binary_minmax_heap(xs)
    valtree = copy(xs)
    for i in length(xs):-1:1
        _minmax_heap_trickle_down!(valtree, i)
    end
    return valtree
end

Base.@propagate_inbounds function _minmax_heap_bubble_up!(A::AbstractVector, i::Integer)
    if on_minlevel(i)
        if i > 1 && A[i] > A[hparent(i)]
            # swap to parent and bubble up max
            tmp = A[i]
            A[i] = A[hparent(i)]
            A[hparent(i)] = tmp
            _minmax_heap_bubble_up!(A, hparent(i), Base.Reverse)
        else
            # bubble up min
            _minmax_heap_bubble_up!(A, i, Base.Forward)
        end

    else
        # max level
        if i > 1 && A[i] < A[hparent(i)]
            # swap to parent and bubble up min
            tmp = A[i]
            A[i] = A[hparent(i)]
            A[hparent(i)] = tmp
            _minmax_heap_bubble_up!(A, hparent(i), Base.Forward)
        else
            # bubble up max
            _minmax_heap_bubble_up!(A, i, Base.Reverse)
        end
    end
end

Base.@propagate_inbounds function _minmax_heap_bubble_up!(
    A::AbstractVector,
    i::Integer,
    o::Base.Ordering,
    x = A[i]
)
    if hasgrandparent(i)
        gparent = hparent(hparent(i))
        if Base.lt(o, x, A[gparent])
            A[i] = A[gparent]
            A[gparent] = x
            _minmax_heap_bubble_up!(A, gparent, o)
        end
    end
end

Base.@propagate_inbounds function _minmax_heap_trickle_down!(A::AbstractVector, i::Integer)
    if on_minlevel(i)
        _minmax_heap_trickle_down!(A, i, Base.Forward)
    else
        _minmax_heap_trickle_down!(A, i, Base.Reverse)
    end
end

Base.@propagate_inbounds function _minmax_heap_trickle_down!(
    A::AbstractVector,
    i::Integer,
    o::Base.Ordering,
    x = A[i],
)

    if haschildren(i, A)
        # get the index of the extremum (min or max) descendant
        extremum = o === Base.Forward ? minimum : maximum
        _, m = extremum((A[j], j) for j in children_and_grandchildren(length(A), i))

        if isgrandchild(m, i)
            if Base.lt(o, A[m], A[i])
                A[i] = A[m]
                A[m] = x
                if Base.lt(o, A[hparent(m)], A[m])
                    t = A[m]
                    A[m] = A[hparent(m)]
                    A[hparent(m)] = t
                end
                _minmax_heap_trickle_down!(A, m, o)
            end
        else
            if Base.lt(o, A[m], A[i])
                A[i] = A[m]
                A[m] = x
            end
        end
    end
end

################################################
#
# utilities
#
################################################

@inline level(i) = floor(Int, log2(i))
@inline lchild(i) = 2*i
@inline rchild(i) = 2*i+1
@inline children(i) = (lchild(i), rchild(i))
@inline hparent(i) = i ÷ 2
@inline on_minlevel(i) = level(i) % 2 == 0
@inline haschildren(i, A) = lchild(i) ≤ length(A)
@inline isgrandchild(j, i) = j > rchild(i)
@inline hasgrandparent(i) = i ≥ 4

"""
    children_and_grandchildren(maxlen, i)

Return the indices of all children and grandchildren of
position `i`.
"""
function children_and_grandchildren(maxlen::T, i::T) where {T <: Integer}
    # Change over DataStructures.jl
    # Make non-allocating by using a lazy filter.
    # Since the number of children and grandchildren is always less than or equal to 6,
    # this ends up being pretty efficient.
    lc = lchild(i)
    rc = rchild(i)
    tup = (lc, lchild(lc), rchild(lc), rc, lchild(rc), rchild(rc))
    return Iterators.filter(x -> x <= maxlen, tup)
end

"""
    is_minmax_heap(h::AbstractVector) -> Bool

Return `true` if `A` is a min-max heap. A min-max heap is a
heap where the minimum element is the root and the maximum
element is a child of the root.
"""
function is_minmax_heap(A::AbstractVector)
    for i in 1:length(A)
        if on_minlevel(i)
            # check that A[i] < children A[i]
            #    and grandchildren A[i]
            for j in children_and_grandchildren(length(A), i)
                A[i] ≤ A[j] || return false
            end
        else
            # max layer
            for j in children_and_grandchildren(length(A), i)
                A[i] ≥ A[j] || return false
            end
        end
    end
    return true
end

################################################
#
# interfaces
#
################################################

Base.length(h::BinaryMinMaxHeap) = length(h.valtree)

Base.isempty(h::BinaryMinMaxHeap) = isempty(h.valtree)

"""
    pop!(h::BinaryMinMaxHeap) = popmin!(h)
"""
@inline Base.pop!(h::BinaryMinMaxHeap) = popmin!(h)

function Base.sizehint!(h::BinaryMinMaxHeap, s::Integer)
    sizehint!(h.valtree, s)
    return h
end

"""
    popmin!(h::BinaryMinMaxHeap) -> min

Remove the minimum value from the heap.
"""
function popmin!(h::BinaryMinMaxHeap)
    valtree = h.valtree
    !isempty(valtree) || throw(ArgumentError("heap must be non-empty"))
    @inbounds begin
        x = valtree[1]
        y = pop!(valtree)
        if !isempty(valtree)
            valtree[1] = y
            _minmax_heap_trickle_down!(valtree, 1)
        end
    end
    return x
end


"""
    popmin!(h::BinaryMinMaxHeap, k::Integer) -> vals

Remove up to the `k` smallest values from the heap.
"""
@inline function popmin!(h::BinaryMinMaxHeap, k::Integer)
    return [popmin!(h) for _ in 1:min(length(h), k)]
end

"""
    popmax!(h::BinaryMinMaxHeap) -> max

Remove the maximum value from the heap.
"""
function popmax!(h::BinaryMinMaxHeap)
    valtree = h.valtree
    !isempty(valtree) || throw(ArgumentError("heap must be non-empty"))
    @inbounds begin
        x, i = maximum(((valtree[j], j) for j in 1:min(length(valtree), 3)))
        y = pop!(valtree)
        if !isempty(valtree) && i <= length(valtree)
            valtree[i] = y
            _minmax_heap_trickle_down!(valtree, i)
        end
    end
    return x
end

"""
    popmax!(h::BinaryMinMaxHeap, k::Integer) -> vals

Remove up to the `k` largest values from the heap.
"""
@inline function popmax!(h::BinaryMinMaxHeap, k::Integer)
    return [popmax!(h) for _ in 1:min(length(h), k)]
end


function Base.push!(h::BinaryMinMaxHeap, v)
    valtree = h.valtree
    push!(valtree, v)
    @inbounds _minmax_heap_bubble_up!(valtree, length(valtree))
end

"""
    first(h::BinaryMinMaxHeap)

Get the first (minimum) of the heap.
"""
@inline Base.first(h::BinaryMinMaxHeap) = minimum(h)

@inline function Base.minimum(h::BinaryMinMaxHeap)
    valtree = h.valtree
    !isempty(h) || throw(ArgumentError("heap must be non-empty"))
    return @inbounds h.valtree[1]
end

@inline function Base.maximum(h::BinaryMinMaxHeap)
    valtree = h.valtree
    !isempty(h) || throw(ArgumentError("heap must be non-empty"))
    return @inbounds maximum(valtree[1:min(end, 3)])
end

Base.empty!(h::BinaryMinMaxHeap) = (empty!(h.valtree); h)


"""
    popall!(h::BinaryMinMaxHeap, ::Ordering = Forward)

Remove and return all the elements of `h` according to
the given ordering. Default is `Forward` (smallest to
largest).
"""
popall!(h::BinaryMinMaxHeap) = popall!(h, Base.Forward)
popall!(h::BinaryMinMaxHeap, ::Base.ForwardOrdering) = popmin!(h, length(h))
popall!(h::BinaryMinMaxHeap, ::Base.ReverseOrdering) = popmax!(h, length(h))

#####
##### Extra Functions
#####

# Consult: https://github.com/JuliaCollections/DataStructures.jl/blob/master/src/heaps/minmax_heap.jl
#
# This should be considered unstable as we're relying on internals of the DataCollections.jl
# package.
#
# Updates to this dependency should be audited to make sure this doesn't change.
Base.@inline function _unsafe_minimum(heap::BinaryMinMaxHeap, default = nothing)
    return isempty(heap) ? default : @inbounds heap.valtree[1]
end

# The `maximum` function in DataStructures.jl allocates because it creates an intermediate
# array. Here, we get around that.
#
# Rely on Julia union splitting `nothing`.
function _unsafe_maximum(heap::BinaryMinMaxHeap, default = nothing)
    isempty(heap) && return default

    # Again, this is relying on the internals of DataStructures.jl
    return maximum(@inbounds heap.valtree[i] for i in 1:min(length(heap), 3))
end

