module PM

export pmmap

import Mmap

const COUNT = Threads.Atomic{Int}(0)
mmap_prefix() = "graphann_mmap_"

# The implementation here is pretty straightforward.
# The atomic count ensures we don't have conflicting file names.
#
# Use a finalizer to remove the backing file when the array gets cleaned up.
function pmmap(::Type{T}, dir::AbstractString, dims::Integer...) where {T}
    num = Threads.atomic_add!(COUNT, 1)
    path = joinpath(dir, join((mmap_prefix(), num)))

    ispath(path) && rm(path; force = true)
    array = Mmap.mmap(path, Array{T,length(dims)}, dims)
    finalizer(array) do _
        # NB: Finalizers in Julia are not allowed to cause Task switches.
        # Wrapping the `rm` function in an `@async` macro defers context switching
        # outside the finalizer.
        #
        # Interpolate the `path` variable with the "$" syntax to copy the string
        # directly into the closure created by this macro.
        @async rm($path; force = true)
    end
    return array
end

end # module
