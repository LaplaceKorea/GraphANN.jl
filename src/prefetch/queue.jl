module Queue

export SemiAtomicQueue

const Atomic = Threads.Atomic

mutable struct SemiAtomicQueue{T}
    buffer::Vector{T}

    # - Writer side variables
    # If `tail != head`, then head always points to a valid, unread entry.
    head::Atomic{Int}
    shadowhead::Int

    # - Read side variables
    # Tail always points to the last value read.
    tail::Atomic{Int}
end

function SemiAtomicQueue{T}(capacity::Integer) where {T}
    buffer = Vector{T}(undef, capacity)
    # Set all pointers equal and at the beginning of the queue.
    # Since `head == tail`, this indicates that there are no available data items.
    head = Atomic{Int}(1)
    shadowhead = 1
    tail = Atomic{Int}(1)
    return SemiAtomicQueue{T}(buffer, head, shadowhead, tail)
end

# For convenience I guess.
const SAQ = SemiAtomicQueue

# Originally, I thought it would be save to use the underlying value directly using `x.value`.
# However, there seems to be a race condition somewhere that would creep up, so now we're
# trying to use the semantic access.
unsafe_get(x::Atomic) = x[]
capacity(x::SAQ) = length(x.buffer)

#####
##### Writer Side API
#####

function Base.push!(queue::SAQ{T}, v::T) where {T}
    # Compute the next shadowhead value
    nextshadow = queue.shadowhead == length(queue.buffer) ? 1 : queue.shadowhead + 1

    # Abort if we're going to collide with the tail pointer.
    # This should always be safe since `queue.tail[]` is ALWAYS conservative of where
    # a reader thread may be reading.
    nextshadow == unsafe_get(queue.tail) && return false

    # Now we know we have room.
    # Simply queue up the item.
    @inbounds queue.buffer[nextshadow] = v
    queue.shadowhead = nextshadow
    return true
end

commit!(queue::SAQ) = Threads.atomic_xchg!(queue.head, queue.shadowhead)

#####
##### Reader Side API
#####

function navailable(queue::SAQ)
    head = unsafe_get(queue.head)
    tail = unsafe_get(queue.tail)
    buffer = queue.buffer

    return (head >= tail) ? (head - tail) : (length(buffer) - tail + head)
end

Base.@propagate_inbounds function consume!(
    v::AbstractVector{T}, start::Integer, queue::SAQ{T}, takemax = nothing
) where {T}
    tail = unsafe_get(queue.tail)
    head = unsafe_get(queue.head)
    buffer = queue.buffer

    # Copy over items
    count = 0
    while tail != head
        tail = (tail == length(buffer)) ? 1 : (tail + 1)
        v[start + count] = buffer[tail]
        count += 1
        count === takemax && break
    end

    # Update tail pointer
    count == 0 || Threads.atomic_xchg!(queue.tail, tail)
    return count
end

end # module
