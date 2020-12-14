module _Threading

export ThreadLocal, getall, allthreads, dynamic_thread, on_threads

#####
##### Thread Local
#####

# A common pattern in Julia programming is to pre-allocate data and mutate it.
# This is especially critical when threading because multi-threaded allocation in Julia
# is pretty darn slow thanks to the garbage collector.
#
# This is a VERY convenient structure that replicates whatever you want for each thread
# and automatically delivers the correct storage bundle when called with `getindex`
# (i.e., the syntax [])

# Thread local storage.
struct ThreadLocal{T}
    values::Vector{T}

    # Inner constructor to resolve ambiguities
    ThreadLocal{T}(values::Vector{T}) where {T} = new{T}(values)
end

# Convenience, wrap around a NamedTuple
ThreadLocal(; kw...) = ThreadLocal((;kw...,))

function ThreadLocal(values::T) where {T}
    return ThreadLocal{T}([deepcopy(values) for _ in 1:Threads.nthreads()])
end

Base.getindex(t::ThreadLocal) = t.values[Threads.threadid()]
Base.setindex!(t::ThreadLocal, v) = (t.values[Threads.threadid()] = v)
getall(t::ThreadLocal) = t.values

#allthreads() = 1:Threads.nthreads()

#####
##### More finegraind thread control
#####

# Collection of threads for work together.
# Allow for partitioning of work.
struct ThreadPool{T <: AbstractVector{<:Integer}}
    threads::T
end

# Forward Methods
@inline Base.eachindex(t::ThreadPool) = eachindex(t.threads)
@inline Base.first(t::ThreadPool) = first(t.threads)
@inline Base.firstindex(t::ThreadPool) = firstindex(t.threads)
@inline Base.length(t::ThreadPool) = length(t.threads)

# Iteration
Base.IteratorSize(::ThreadPool) = Base.HasLength()
Base.iterate(t::ThreadPool, s...) = iterate(t.threads, s...)

# Ref:
# https://github.com/oschulz/ParallelProcessingTools.jl/blob/6a354b4ac7e90942cfe1d766d739306852acb0db/src/onthreads.jl#L14
# Schedules a task on a given thread.
function _schedule(t::Task, tid)
    @assert !istaskstarted(t)
    t.sticky = true
    ccall(:jl_set_task_tid, Cvoid, (Any, Cint), t, tid-1)
    schedule(t)
    return t
end

function on_threads(
    func::F,
    pool::ThreadPool,
    wait = true,
) where {F}
    tasks = Vector{Task}(undef, length(eachindex(pool)))
    for tid in pool
        i = firstindex(tasks) + (tid - first(pool))
        tasks[i] = _schedule(Task(func), tid)
    end
    wait && foreach(Base.wait, tasks)
    return nothing
end

# `allthreads()` returns a ThreadPool for (believe it or not) all threads!
allthreads() = ThreadPool(1:Threads.nthreads())

#####
##### Dynamic
#####

# Default to using all threads
dynamic_thread(f::F, args...) where {F} = dynamic_thread(f, allthreads(), args...)

# Julia doesn't implement dynamic threading yet ...
# So we have to do it on our own.
#
# Fortunately, it's quite easy!
function dynamic_thread(f::F, pool::ThreadPool, domain, worksize = 1) where {F}
    count = Threads.Atomic{Int}(1)
    len = length(domain)
    on_threads(pool) do
        while true
            k = Threads.atomic_add!(count, 1)
            start = worksize * (k - 1) + 1
            start > len && break

            stop = min(worksize * k, len)
            for i in start:stop
                f(@inbounds domain[i])
            end
        end
    end
end

end # module
