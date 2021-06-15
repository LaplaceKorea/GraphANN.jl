# This is a partially applied version of the full allocator above.
# Use it like `f = pmallocator("path/to/dir")` to construct a function `f` that will
# have the same signature as the `stdallocator` above.
struct PMAllocator
    dir::String
end

(f::PMAllocator)(type, dims...) = pmmap(type, f.dir, dims...; rmfile = true)

"""
    pmallocator(dir::AbstractString)

Construct an allocator that will when invoked will memory map files in directory `dir`
to fullfill the allocation.

When `dir` points to a direct access (DAX) file system backed by persistent memory, then
the memory returned by invoking `pmallocator` will reside in persistent memory.

Memory mapped files will begin with the prefix `graphann_mmap` and will be deleted when
the corresponding memory is garbage collected.

# Example
```jldoctest
julia> allocator = GraphANN.pmallocator(pwd());

julia> A = allocator(Int64, 2, 2)
2×2 Matrix{Int64}:
 0  0
 0  0
```
"""
pmallocator(dir::AbstractString) = PMAllocator(dir)

const COUNT = Threads.Atomic{Int}(0)
mmap_prefix() = "graphann_mmap_"

# The implementation here is pretty straightforward.
# The atomic count ensures we don't have conflicting file names.
#
# Use a finalizer to remove the backing file when the array gets cleaned up.
function pmmap(::Type{T}, dir::AbstractString, dims::Integer...; rmfile = true) where {T}
    num = Threads.atomic_add!(COUNT, 1)
    path = joinpath(dir, join((mmap_prefix(), num)))

    ispath(path) && rm(path; force = true)
    array = Mmap.mmap(path, Array{T,length(dims)}, convert.(Int64, dims))
    if rmfile
        finalizer(array) do _
            # NB: Finalizers in Julia are not allowed to cause Task switches.
            # Wrapping the `rm` function in an `@async` macro defers context switching
            # outside the finalizer.
            #
            # Interpolate the `path` variable with the "$" syntax to copy the string
            # directly into the closure created by this macro.
            @async rm($path; force = true)
        end
    end
    return array
end

