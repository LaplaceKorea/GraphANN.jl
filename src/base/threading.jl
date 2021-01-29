#####
##### More finegrained thread control
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

# `allthreads()` returns a ThreadPool for (believe it or not) all threads!
allthreads() = ThreadPool(1:Threads.nthreads())

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

# If launching non-blocking tasks, it's helpful to be able to retrieve the tasks in case
# an error happens.
#
# Thus, we make `on_threads` return an `OnThreads` instances that wraps all the launched
# tasks.
#
# This allows us to wait on the tasks and potentially catch errors.
struct TaskHandle
    tasks::Vector{Task}
end
Base.wait(t::TaskHandle) = foreach(Base.wait, t.tasks)
Base.length(t::TaskHandle) = length(t.tasks)

function on_threads(
    func::F,
    pool::ThreadPool,
    wait::Bool = true,
) where {F}
    tasks = Vector{Task}(undef, length(eachindex(pool)))
    for tid in pool
        i = firstindex(tasks) + (tid - first(pool))
        tasks[i] = _schedule(Task(func), tid)
    end
    handle = TaskHandle(tasks)
    wait && Base.wait(handle)
    return handle
end

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
struct ThreadLocal{T,U}
    # When used on conjunction with a ThreadPool, we need to make this a dictionary
    # because we aren't guarenteed that thread id's start at 1.
    values::Dict{Int64,T}
    pool::ThreadPool{U}

    # Inner constructor to resolve ambiguities
    function ThreadLocal{T}(values::Dict{Int64,T}, pool::ThreadPool{U}) where {T, U}
        return new{T, U}(values, pool)
    end
end

# Convenience, wrap around a NamedTuple
# Need to define a few methods to get around ambiguities.
ThreadLocal(; kw...) = ThreadLocal(allthreads(), (;kw...,))
ThreadLocal(pool::ThreadPool; kw...) = ThreadLocal(pool, (;kw...,))
ThreadLocal(values) = ThreadLocal(allthreads(), values)

function ThreadLocal(pool::ThreadPool, values::T) where {T}
    return ThreadLocal{T}(Dict(tid => deepcopy(values) for tid in pool), pool)
end

Base.getindex(t::ThreadLocal, i::Integer = Threads.threadid()) = t.values[i]
Base.setindex!(t::ThreadLocal, v) = (t.values[Threads.threadid()] = v)
getall(t::ThreadLocal) = collect(values(t.values))
getpool(t::ThreadLocal) = t.pool
