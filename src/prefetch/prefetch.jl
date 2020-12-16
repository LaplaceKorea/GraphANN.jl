#####
##### Prefetcher Implementation
#####

module _Prefetcher

import UnPack: @unpack, @pack!
using .._Base

# Many of the pieces that make up the prefetcher implementations are wrapped in inner
# modules.
#
# This is done to help avoid polluting the main namespace.
include("atomic.jl")
include("queue.jl"); import .Queue: SemiAtomicQueue, commit!, consume!

mutable struct Staging{T}
    # One queue for each query thread.
    # Keep track of a current index so we can access this vector in a round-robin manner.
    queues::Vector{SemiAtomicQueue{T}}
    index::Int

    # The staging area is protected by lock so only one prefetcher
    # thread can access it at a time.
    lock::ReentrantLock
end

function Staging(queues::Vector{SemiAtomicQueue{T}}) where {T}
    return Staging{T}(queues, 1, ReentrantLock())
end

function acquire!(x::Staging{T}, dest::Vector{T}, num::Integer) where {T}
    # Prepare destination
    resize!(dest, num)

    # Global lockitems_collected.
    # Access work items in the queues in a round robin manner.
    # Go until `num` tokens have been retrieved or all queues have been visited.
    items_collected = 0
    Base.@lock x.lock begin
        @unpack queues, index = x

        num_queues = length(queues)
        queues_visited = 0
        while true
            count = consume!(
                dest,
                items_collected + 1,
                queues[index],
                num - items_collected,
            )

            # Update counters
            items_collected += count
            queues_visited += 1
            index = (index == num_queues) ? 1 : (index + 1)

            (items_collected >= num || queues_visited == num_queues) && break
        end

        @pack! x = index
    end
    resize!(dest, items_collected)
    return nothing
end

#####
##### Prefetcher
#####

struct PrefetchRunner{S <: Staging, T}
    # Staging area where we get work items from.
    staging::S
    # Pre-allocated temporary data-structures.
    worklist::Vector{T}
    tryfor::Int

    # Stop Token
    stop::Ref{Bool}
end

stop!(prefetcher::PrefetchRunner) = (prefetcher.stop[] = true)
reset!(prefetcher::PrefetchRunner) = (prefetcher.stop[] = false)

function prefetch_loop(prefetcher::PrefetchRunner, data)
    @unpack staging, worklist, stop, tryfor = prefetcher
    while !stop[]
        # Gather incoming ids
        acquire!(staging, worklist, tryfor)

        # Try to prefetch each data item into the LLC
        # Hopefully doesn't cause eviction if the data is in someone else's L2 ...
        for index in worklist
            prefetch(data, index, prefetch_llc)
        end
    end
end

#####
##### Ensemple of Prefetchers
#####

mutable struct Prefetcher{T, S <: Staging, U <: Integer}
    thread_pool::ThreadPool{T}
    staging::S

    # Signals to runners - when `true`, stop prefetching.
    stop_signals::Vector{Ref{Bool}}
    worklists::Vector{Vector{U}}

    # Handle to the tasks that are currently running.
    # Maintain the following invariants:
    # - `handle === nothing` only if we're sure no tasks spawned by the prefetcher are
    # running.
    # - Otherwise, tasks must be reachable until they are stopped or killed.
    # - Once stopped of killed, this field may be reset to `nothing`.
    tasks::Union{Nothing, TaskHandle}
end

stop!(p::Prefetcher) = foreach(i -> i[] = true, p.stop_signals)
reset!(p::Prefetcher) = foreach(i -> i[] = false, p.stop_signals)

function spin(prefetcher::Prefetcher, data; tryfor = 100)
    @unpack thread_pool, staging, worklists, stop_signals = prefetcher
    reset!(prefetcher)
    handle = on_threads(thread_pool, false) do
        # Obtain thread local storage
        tid = Threads.threadid()
        worklist = worklists[tid]
        stop_signal = stop_signals[tid]

        runner = PrefetchRunner(staging, worklist, tryfor, stop_signal)
        prefetch_loop(runner, data)
    end
    @pack! prefetcher = handle
    return nothing
end

end # module
