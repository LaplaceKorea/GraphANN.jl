# The indirection table maintains a vector of aligned pointers to items of type `T`.
# Intenerally, these values are stored as UInt64.
#
# This allows us to use the lower bits of these values as locks for the corresponding
# entrties.
const NUM_SAVED_BITS = 1
const POINTER_MASK = (one(UInt64) << NUM_SAVED_BITS) - one(UInt64)

struct IndirectionTable{T}
    # Keep a reference to the wrapped base datastructure to keep it from getting GC'd.
    base::Vector{T}
    pointers::Vector{UInt64}
end

# Construct an indirection table wrapping a vector.
function wrapvector(v::Vector{T}) where {T}
    return IndirectionTable{T}(v, [pointer(v, i) for i in eachindex(v)])
end

Base.pointer(A::IndirectionTable, i) = pointer(A.pointers, i)

function Base.get(A::IndirectionTable{T}, i::Integer) where {T}
    return Ptr{T}(pointer(A, i) & ~POINTER_MASK)
end

# Add a specialization point.
@inline loadhook(ptr::Ptr) = unsafe_load(ptr)
Base.getindex(A::IndirectionTable, i::Integer) = loadhook(get(A, i))

function Base.trylock(A::IndirectionTable, i::Integer)
    @boundscheck checkbounds(A.pointers, i)

    # Try to set the lower bit of the pointer atomically.
    # The function `unsafe_atomic_or!` returns the old value.
    # If the lower bit of the returned value is zero, then we've acquired the lock.
    # Otherwise, someone else holds the lock.
    val = unsafe_atomic_or!(pointer(A, i), POINTER_MASK)
    return iszero(val & POINTER_MASK)
end

function Base.unlock(A::IndirectionTable, i::Integer)
    @boundscheck checkbounds(A.pointers, i)

    # Unconditionally clear the bottom of the pointer.
    unsafe_atomic_nand!(pointer(A, i), POINTER_MASK)
    return nothing
end

# Don't make the following two functions atomic.
# Require memory fence after all sets/resets are complete.
function set!(A::IndirectionTable{T}, ptr::Ptr{T}, i) where {T}
    A.pointers[i] = UInt64(ptr) | POINTER_MASK
end

reset!(A::IndirectionTable, i) = (A.pointers[i] = pointer(A.v, i))
