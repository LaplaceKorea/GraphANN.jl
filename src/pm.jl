module PM

export pmmap

import Mmap

const COUNT = Threads.Atomic{Int}(0)
mmap_prefix() = "graphann_mmap_"

function pmmap(::Type{T}, dir::AbstractString, len::Integer) where {T}
    num = Threads.atomic_add!(COUNT, 1)
    path = joinpath(dir, join((mmap_prefix(), num)))

    array = Mmap.mmap(path, Vector{T}, len, 0)
    finalizer(array) do _
        rm(path; force = true)
    end
    return array
end

end # module
