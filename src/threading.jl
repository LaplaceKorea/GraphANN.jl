#####
##### Thread Local
#####

# A common patternin Julia programming is to pre-allocate data and mutate it.
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
getall(t::ThreadLocal) = t.values

allthreads() = 1:Threads.nthreads()

#####
##### Dynamic
#####

# Julia doesn't implement dynamic threading yet ...
# So we have to do it on our own.
#
# Fortunately, it's quite easy!
@static if ENABLE_THREADING
    function dynamic_thread(f::F, domain, worksize = 1) where {F}
        count = Threads.Atomic{Int}(1)
        len = length(domain)
        Threads.@threads for j in allthreads()
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
else
    dynamic_thread(f::F, domain, x...) where {F} = map(f, domain)
end # @static if

