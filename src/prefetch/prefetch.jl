#####
##### Prefetcher Implementation
#####

module _Prefetcher

import UnPack: @unpack, @pack!

# Many of the pieces that make up the prefetcher implementations are wrapped in inner
# modules.
#
# This is done to help avoid polluting the main namespace.
include("atomic.jl")
include("indirection.jl")
include("queue.jl"); import .Queue: SemiAtomicQueue

mutable struct Staging{T}
    # One queue for each query thread.
    # Keep track of a current index so we can access this vector in a round-robin manner.
    queues::Vector{SemiAtomicQueue{T}}
    index::Int

    # The staging area is protected by lock so only one prefetcher
    # thread can access it at a time.
    lock::ReentrantLock
end

function acquire!(x::Staging{T}, dest::Vector{T}, num::Integer) where {T}
    # Prepare destination
    resize!(dest, num)

    # Global lock.
    # Access work items in the queues in a round robin manner.
    # Go until `num` tokens have been retrieved or all queues have been visited.
    Base.@lock x.lock begin
        @unpack queues, index = x

        num_queues = length(queues)
        visited = 0
        start = 1
        while true
            count = consume!(dest, start, queues[index], num + 1 - start)

            # Update counters
            start += count
            visited += 1
            index = (index == num_queues) ? 1 : (index + 1)

            (start >= num || visited == num_queues) && break
        end

        @pack! x = index
    end
    return nothing
end

#####
##### Prefetcher
#####

struct Prefetcher{T, U <: Unsigned, S <: Staging}
    # Local DRAM cache
    cache::Vector{T}
    # Position-wise correlation between ids in the local cache and global ids.
    # Use a sentinal value of 0 to indicate that a slot is not filled.
    ids::Vector{U}
    # Staging area where we get work items from.
    staging::S

    # -- pre-allocated temporary data-structures.
    prefetch_worklist::Vector{U}
    free_worklist::Vector{U}
end

end # module
