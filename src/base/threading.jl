#####
##### More finegrained thread control
#####

"""
    ThreadPool

Collection of thread-ids that can be passed to [`_Base.on_threads`](@ref) to launch
tasks onto specific threads.
"""
struct ThreadPool{T<:AbstractVector{<:Integer}}
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

"""
    allthreads()

Return a [`_Base.ThreadPool`](@ref) containing all valid thread-ids for the current
Julia session.
"""
allthreads() = ThreadPool(Base.OneTo(Threads.nthreads()))

# Ref:
# https://github.com/oschulz/ParallelProcessingTools.jl/blob/6a354b4ac7e90942cfe1d766d739306852acb0db/src/onthreads.jl#L14
# Schedules a task on a given thread.
function _schedule(t::Task, tid)
    @assert !istaskstarted(t)
    t.sticky = true
    ccall(:jl_set_task_tid, Cvoid, (Any, Cint), t, tid - 1)
    schedule(t)
    return t
end

# If launching non-blocking tasks, it's helpful to be able to retrieve the tasks in case
# an error happens.
"""
    TaskHandle

Reference to a group of tasks launched by [`_Base.on_threads`](@ref).
Can pass to `Base.wait` to block execution until all tasks have completed.
"""
struct TaskHandle
    tasks::Vector{Task}
end
Base.wait(t::TaskHandle) = foreach(Base.wait, t.tasks)
Base.length(t::TaskHandle) = length(t.tasks)

"""
    on_threads(f, threadpool::ThreadPool, [wait = true]) -> TaskHandle

Launch a task for function `f` for each thread in `threadpool`.
Return a [`TaskHandle`](@ref) for the launched tasks.

If `wait = true`, then execution is blocked until all launched tasks complete.
"""
function on_threads(func::F, pool::ThreadPool, wait::Bool = true) where {F}
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

"""
    single_thread(f, domain)

Apply `f` to each element in `domain` using a single thread.

# Example
```julia
julia> x = [1,2,3]

julia> GraphANN.single_thread(println, x)
1
2
3
```
"""
single_thread(f::F, domain, args...) where {F} = foreach(f, domain)
single_thread(f::F, ::ThreadPool, domain, args...) where {F} = single_thread(f, domain)

"""
    dynamic_thread(f, [threadpool], domain, [worksize])

Apply `f` to each element of `domain` using dynamic load balancing among the threads in
`threadpool`. If `threadpool` is not given, it defaults to [`_Base.allthreads()`](@ref).
No guarentees are made about the order of execution.

Optional argument `worksize` controls the granularity of the load balancing.

# Example
```julia
julia> lock = ReentrantLock();

julia> GraphANN.dynamic_thread(1:10) do i
    Base.@lock lock println(i)
end
1
5
6
7
8
9
10
4
3
2
```
"""
dynamic_thread(f::F, args...) where {F} = dynamic_thread(f, allthreads(), args...)
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
"""
    ThreadLocal{T}

Create an instance of type `T` for each thread in a [`ThreadPool`](@ref).
The thread pool defaults to [`allthreads()`](@ref) but may be customized.

## Examples

The examples below assume that Julia has been started with 4 threads (JULIA_NUM_THREADS = 4)>
```julia
# Basic constructor - creates a copy of the argument for each thread.
juila> x = GraphANN.ThreadLocal(Int[]);

# Access all values using `getall`.
julia> GraphANN.getall(x)
4-element Vector{Vector{Int64}}:
 []
 []
 []
 []

# A thread can access its local version using `getindex`.
julia> y = x[]
Int64[]

# Objects are deepcopied, so each thead by default gets a unique object.
julia> push!(y, 10); GraphANN.getall(x)
4-element Vector{Vector{Int64}}:
 [10]
 []
 []
 []
```
As can be seen in the above example, each object replicated by the `ThreadLocal` constructor
is replicated using `deepcopy`. To customize this behavior, extend `GraphANN.threadcopy`
for the desired type to implement the desired copying behavior.

If multiple different structs are needed for each thread, than a `ThreadLocal` can also
be constructed using keywords, which will be converted into a `NamedTuple`.
```julia
julia> x = GraphANN.ThreadLocal(; a = 1, b = Int64[]);

julia> GraphANN.getall(x)
4-element Vector{NamedTuple{(:a, :b), Tuple{Int64, Vector{Int64}}}}:
 (a = 1, b = [])
 (a = 1, b = [])
 (a = 1, b = [])
 (a = 1, b = [])

julia> x[]
(a = 1, b = Int64[])
```
Different thread pools can also be provided as an optional first argument.
```julia
julia> x = GraphANN.ThreadLocal(GraphANN.ThreadPool([2,4,5]); a = 1, b = Int64[]);

julia> x[2]
(a = 1, b = Int64[])

# Note - since thread 1 is not in the thread pool, it cannot access `x` using the normal
# `getindex` method.
julia> x[]
ERROR: KeyError: key 1 not found
[...]
```
"""
struct ThreadLocal{T,U}
    # When used on conjunction with a ThreadPool, we need a dictionary to translate from
    # thread id to index because we aren't guarenteed that thread id's start at 1.
    #
    # However, in the case where the `ThreadPool` wraps a `Base.OneTo`, we can elide the
    # dictionary check.
    values::Vector{T}
    pool::ThreadPool{U}
    translation::Dict{Int64,Int64}

    # Inner constructor to resolve ambiguities
    function ThreadLocal{T}(values::Vector{T}, pool::ThreadPool{U}) where {T,U}
        translation = Dict(i => j for (j, i) in enumerate(pool))
        return new{T,U}(values, pool, translation)
    end

    function ThreadLocal{T}(
        values::Vector{T}, pool::ThreadPool{U}, translation::Dict{Int,Int}
    ) where {T,U}
        return new{T,U}(values, pool, translation)
    end
end

# Many algorithms are designed to either take thread local of some struct for storage when
# using a multi-threaded implementation, or just the simple data structure itself if
# performing single threaded operation.
const MaybeThreadLocal{T} = Union{T,ThreadLocal{<:T}}

# Convenience, wrap around a NamedTuple
# Need to define a few methods to get around ambiguities.
ThreadLocal(; kw...) = ThreadLocal(allthreads(), (; kw...))
ThreadLocal(pool::ThreadPool; kw...) = ThreadLocal(pool, (; kw...))
ThreadLocal(values) = ThreadLocal(allthreads(), values)

threadcopy(x) = deepcopy(x)
@generated function threadcopy(x::NamedTuple{names}) where {names}
    exprs = [:(threadcopy(x.$name)) for name in names]
    return :(NamedTuple{names}(($(exprs...),)))
end

function ThreadLocal(pool::ThreadPool, values::T) where {T}
    # Perform each copy on its own thread to ensure that if we're doing any kind of NUMA
    # scheduling that thread-local memory gets allocated on the correct numa node.
    copies = Vector{T}()
    foreach(pool) do tid
        on_threads(ThreadPool(tid:tid)) do
            push!(copies, threadcopy(values))
        end
    end
    return ThreadLocal{T}(copies, pool)
end

Base.getindex(t::ThreadLocal, i::Integer = Threads.threadid()) = t.values[t.translation[i]]
Base.setindex!(t::ThreadLocal, v) = (t.values[t.translation[Threads.threadid()]] = v)

# Optimized for default case.
function Base.getindex(t::ThreadLocal{<:Any,<:Base.OneTo}, i::Integer = Threads.threadid())
    return t.values[i]
end

function Base.setindex!(
    t::ThreadLocal{<:Any,<:Base.OneTo}, v, i::Integer = Threads.threadid()
)
    return t.values[i] = v
end

"""
    getall(x::ThreadLocal)

Return all thread local datastructures in `x`.
"""
getall(t::ThreadLocal) = t.values
# Often, a `NamedTuple` can be passed to routines instead of a `ThreadLocal` if the routine
# is using a single thread. Wrap that tuple inside of another tuple so iteration works
# as expected.
getall(nt::NamedTuple) = (nt,)

"""
    getpool(x::ThreadLocal)

Return the [`ThreadPool`](@ref) for `x`.
"""
getpool(t::ThreadLocal) = t.pool

"""
    getlocal(x)

If `x` is a [`ThreadLocal`](@ref), then return the local data structure for the current
thread. Otherwise, just return `x`.
"""
getlocal(x::ThreadLocal) = x[]
getlocal(x) = x

