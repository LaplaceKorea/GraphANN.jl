#####
##### Prefetcher Implementation
#####

module _Prefetcher

export commit!, consume!

# local deps
using .._Base
using .._Graphs

# deps
import LightGraphs
import UnPack: @unpack, @pack!

include("queue.jl"); import .Queue: SemiAtomicQueue, commit!, consume!

# Behaves like a metagraph, but provides prefetching as well!
struct PrefetchedMeta{D, G, P}
    data::D
    graph::G
    prefetcher::P
end

getqueue(A::PrefetchedMeta) = getqueue(A.prefetcher)
Base.getindex(A::PrefetchedMeta, i...) = getindex(A.meta, i...)

function start(A::PrefetchedMeta; kw...)
    if isrunning(A.prefetcher)
        printlnstyled("Not relaunching prefetcher."; color = :green, bold = true)
    else
        start(A.prefetcher, MetaGraph(A.graph, A.data); kw...)
    end
    return nothing
end

stop(A::PrefetchedMeta) = stop!(A.prefetcher)

function prefetch_wrap(
    meta::MetaGraph,
    query_pool::ThreadPool,
    prefetch_pool::ThreadPool;
    queue_capacity = 1000,
    int_type::Type{U} = UInt32,
) where {U}
    @unpack data, graph = meta
    # Construct queues for each thread in `query_pool`, then construct a staging area
    # that serves as the destination for all these queues.
    queues = Dict(i => SemiAtomicQueue{U}(queue_capacity) for i in query_pool)
    staging = Staging(queues)

    prefetcher = Prefetcher{U}(prefetch_pool, staging)
    return PrefetchedMeta(data, graph, prefetcher)
end

#####
##### Staging
#####

# The staging area is a struct that synchronized between the query threads and the prefetch
# threads.
#
# Query threads write to the staging area using queues, one per query thread.
# Furthermore, the query threads own the write side of these queues, so are free to write
# whenever they want.
#
# On the other hand, all prefetch threads share the read side of the queues, which is
# protected by a single lock that much be held in order to modify the read state of any
# queue.
mutable struct Staging{T}
    # One queue for each query thread.
    # Keep track of a current index so we can access this vector in a round-robin manner.
    queues::Dict{Int,SemiAtomicQueue{T}}
    queue_iter::Vector{SemiAtomicQueue{T}}
    index::Int

    # The staging area is protected by lock so only one prefetcher
    # thread can access it at a time.
    lock::ReentrantLock
end

function Staging(queues::Dict{Int,SemiAtomicQueue{T}}) where {T}
    queue_iter = collect(values(queues))
    return Staging{T}(queues, queue_iter, 1, ReentrantLock())
end

getqueue(x::Staging) = x.queues[Threads.threadid()]

function acquire!(x::Staging{T}, dest::Vector{T}, num::Integer) where {T}
    # Prepare destination
    resize!(dest, num)

    # Global lockitems_collected.
    # Access work items in the queues in a round robin manner.
    # Go until `num` tokens have been retrieved or all queues have been visited.
    items_collected = 0
    Base.@lock x.lock begin
        @unpack queue_iter, index = x

        num_queues = length(queue_iter)
        queues_visited = 0
        while true
            count = consume!(
                dest,
                items_collected + 1,
                queue_iter[index],
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

function prefetch_loop(prefetcher::PrefetchRunner, meta::MetaGraph)
    @unpack staging, worklist, stop, tryfor = prefetcher
    @unpack data, graph = meta

    while !stop[]
        # Gather incoming ids
        acquire!(staging, worklist, tryfor)

        # Try to prefetch each data item into the LLC
        # Hopefully doesn't cause eviction if the data is in someone else's L2 ...
        for u in worklist, v in LightGraphs.outneighbors(graph, u)
            prefetch(data, v, prefetch_llc)
        end
    end
    return nothing
end

#####
##### Ensemple of Prefetchers
#####

mutable struct Prefetcher{U <: Integer, T, S <: Staging}
    thread_pool::ThreadPool{T}
    staging::S

    # Signals to runners - when `true`, stop prefetching.
    stop_signals::Dict{Int, Base.RefValue{Bool}}
    worklists::Dict{Int, Vector{U}}

    # Handle to the tasks that are currently running.
    # Maintain the following invariants:
    # - `handle === nothing` only if we're sure no tasks spawned by the prefetcher are
    # running.
    # - Otherwise, tasks must be reachable until they are stopped or killed.
    # - Once stopped of killed, this field may be reset to `nothing`.
    tasks::Union{Nothing, TaskHandle}
end

function Prefetcher{U}(thread_pool::ThreadPool, staging::Staging) where {U}
    stop_signals = Dict(i => Ref(false) for i in thread_pool)
    worklists = Dict(i => U[] for i in thread_pool)

    return Prefetcher(
        thread_pool,
        staging,
        stop_signals,
        worklists,
        nothing,
    )
end

isrunning(p::Prefetcher) = (p.tasks !== nothing)
getqueue(p::Prefetcher) = getqueue(p.staging)
function stop!(p::Prefetcher)
    printlnstyled("Sending Stop Signals"; color = :yellow, bold = true)
    foreach(i -> i[] = true, values(p.stop_signals))
    wait(values(p.tasks))
    printlnstyled("Tasks Halted"; color = :green, bold = true)
    reset!(p)
    p.tasks = nothing
end
reset!(p::Prefetcher) = foreach(i -> i[] = false, values(p.stop_signals))

function start(prefetcher::Prefetcher, meta; tryfor = 100)
    @unpack thread_pool, staging, worklists, stop_signals = prefetcher
    reset!(prefetcher)
    tasks = on_threads(thread_pool, false) do
        # Obtain thread local storage
        tid = Threads.threadid()
        worklist = worklists[tid]
        stop_signal = stop_signals[tid]

        runner = PrefetchRunner(staging, worklist, tryfor, stop_signal)
        prefetch_loop(runner, meta)
    end
    @pack! prefetcher = tasks
    return nothing
end

end # module
