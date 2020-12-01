#####
##### Prefetcher Implementation
#####

module _Prefetcher

# Many of the pieces that make up the prefetcher implementations are wrapped in inner
# modules.
#
# This is done to help avoid polluting the main namespace.
include("atomic.jl")
include("indirection.jl")
include("queue.jl"); import .Queue: SemiAtomicQueue

struct Staging{T}
    # One queue for each query thread.
    queues::Vector{SemiAtomicQueue{T}}
    # The staging area is protected by lock so only one prefetcher
    # thread can access it at a time.
    lock::ReentrantLock
end

end # module
