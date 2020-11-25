#####
##### Prefetcher Implementation
#####

include("atomic.jl"); import .Atomics: unsafe_atomic_cas!, unsafe_atomic_or!, unsafe_atomic_nand!
include("queue.jl"); import .Queue: SemiAtomicQueue

struct Staging{T}
    # One queue for each query thread.
    queues::Vector{SemiAtomicQueue{T}}
    # The staging area is protected by lock so only one prefetcher
    # thread can access it at a time.
    lock::Threads.SpinLock
end


