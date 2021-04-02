# Preprocess a file with the expected element type.
ismatch(r::Regex, s) = (match(r, s) !== nothing)

# Auto detect format based on file extension.
# The return type of this function is not type-stable ... but that's okay, because it
# shouldn't be called in an performance critical code.
function load_vecs(file; kw...)
    # This will turn "/file/path/somefile.ivecs" into just ".ivecs".
    ext = last(splitext(file))

    # Is the extensions what we wanted it to be?
    if !ismatch(r"^\.(i|b|f)vecs", ext)
        throw(ArgumentError("Unknown file extension: \"$ext\"!"))
    end

    # What is the data type?
    datatype_char = ext[2]
    typemap = Dict('b' => UInt8, 'i' => UInt32, 'f' => Float32)
    return load_vecs(typemap[datatype_char], file; kw...)
end

# Overload points for customizing loading.
vecs_read_type(::Type{T}) where {T} = T
vecs_read_type(::Type{SVector{N,T}}) where {N,T} = T

vecs_reshape(::Type, v, dim) = reshape(v, convert(Int, dim), :)
vecs_reshape(::Type{<:SVector}, v, dim) = v

function addto!(x::AbstractVector{T}, i, v::AbstractVector{T}) where {T}
    copyto!(x, i, v, 1, length(v))
    return length(v)
end

function addto!(v::Vector{SVector{N,T}}, index, buf::AbstractVector{T}) where {N,T}
    length(buf) == N || error("Lenght of buffer is incorrect!")
    ptr = Ptr{T}(pointer(v, index))
    unsafe_copyto!(ptr, pointer(buf), N)
    return 1
end

"""
    load_vecs(::Type{T}, file::AbstractString; kw...)

Parse the [`vecs`](http://corpus-texmex.irisa.fr/) into a
* `Matrix{T}` if `T` is a scalar such as `UInt8`, `Float32`, or `UInt32`. Vectors are
    loaded as columns of the returned matrix.

* `Vector{T}` if `T` is an `SVector{N,T}`. Throws an error if `N` doesn't match the parsed
    vector size extracted from the `vecs` file itself.

## Keyword Arguments

* `maxlines::Integer` - If provided, read at most `maxlines` vectors from the dataset.
    Default: `nothing` (whole data file).
* `allocator` - Allocator to use for memory. Default: [`stdallocator`](@ref)
* `groundtruth::Bool` - Set to `true` if loading a groundtruth file. Since Julia is
    index-1 while vecs files tend to be index-0, `groundtruth = true` will essentially
    increment every read value by 1.

## Example

```jldoctest; filter = r"\\[.*\\]"
julia> path = joinpath(GraphANN.VECSDIR, "siftsmall_base.fvecs");

julia> mat = GraphANN.load_vecs(Float32, path)
128×10000 Matrix{Float32}:
  0.0   14.0    0.0   12.0   1.0   48.0  …    6.0  38.0  48.0    0.0  14.0
 16.0   35.0    1.0   47.0   1.0   69.0      40.0  64.0   1.0    0.0   2.0
 35.0   19.0    5.0   14.0   0.0    9.0      68.0  28.0   0.0    0.0   0.0
  5.0   20.0    3.0   25.0   0.0    6.0       3.0   0.0   0.0    5.0   0.0
 32.0    3.0   44.0    2.0  14.0    2.0       2.0   0.0   0.0   43.0   0.0
 31.0    1.0   40.0    3.0  16.0    3.0  …    1.0   0.0   1.0  100.0   2.0
 14.0   13.0   20.0    4.0  30.0    7.0       0.0   0.0  15.0   54.0  42.0
 10.0   11.0   14.0    7.0  50.0   25.0       0.0   5.0  64.0   12.0  55.0
 11.0   16.0   10.0   14.0   2.0   64.0     114.0  10.0  69.0    4.0   9.0
 78.0  119.0  100.0  122.0  40.0  130.0      45.0  98.0  11.0    0.0   1.0
  ⋮                                 ⋮    ⋱    ⋮
 41.0   52.0   31.0   43.0  48.0   31.0       5.0  49.0  22.0   46.0   7.0
  0.0   15.0    0.0   15.0   0.0   20.0  …    9.0   0.0   1.0   31.0  50.0
  0.0    2.0    1.0    1.0   0.0    8.0       6.0   0.0   0.0    0.0  36.0
  2.0    0.0    6.0    0.0   2.0    2.0       0.0   3.0   0.0    1.0  15.0
  8.0    0.0   10.0    0.0   0.0    0.0       0.0  11.0   0.0    2.0  11.0
 19.0    0.0   12.0    0.0   1.0    0.0       3.0   6.0   0.0   16.0   1.0
 25.0   11.0    4.0   27.0   4.0   30.0  …   94.0   2.0  22.0    3.0   0.0
 23.0   21.0   23.0   29.0  28.0   26.0      36.0   1.0  62.0    3.0   0.0
  1.0   33.0   10.0   21.0  34.0    4.0       2.0   3.0  18.0   11.0   7.0


julia> vec = GraphANN.load_vecs(GraphANN.SVector{128,Float32}, path)
10000-element Vector{StaticArrays.SVector{128, Float32}}:
 [0.0, 16.0, 35.0, ... 25.0, 23.0, 1.0]
 [14.0, 35.0, 19.0, ... 11.0, 21.0, 33.0]
 [0.0, 1.0, 5.0, ... 4.0, 23.0, 10.0]
 [12.0, 47.0, 14.0, ... 27.0, 29.0, 21.0]
 [1.0, 1.0, 0.0, ... 4.0, 28.0, 34.0]
 [48.0, 69.0, 9.0, ... 30.0, 26.0, 4.0]
 [0.0, 42.0, 55.0, ... 45.0, 11.0, 2.0]
 [16.0, 36.0, 10.0, ... 3.0, 7.0, 41.0]
 [8.0, 35.0, 11.0, ... 17.0, 4.0, 7.0]
 [21.0, 13.0, 18.0, ... 51.0, 36.0, 3.0]
 ⋮
 [0.0, 0.0, 0.0, ... 0.0, 0.0, 2.0]
 [2.0, 13.0, 17.0, ... 2.0, 3.0, 5.0]
 [43.0, 12.0, 0.0, ... 0.0, 3.0, 17.0]
 [5.0, 5.0, 10.0, ... 1.0, 1.0, 0.0]
 [6.0, 40.0, 68.0, ... 94.0, 36.0, 2.0]
 [38.0, 64.0, 28.0, ... 2.0, 1.0, 3.0]
 [48.0, 1.0, 0.0, ... 22.0, 62.0, 18.0]
 [0.0, 0.0, 0.0, ... 3.0, 3.0, 11.0]
 [14.0, 2.0, 0.0, ... 0.0, 0.0, 7.0]
```
"""
function load_vecs(
    ::Type{T}, file; maxlines = nothing, allocator = stdallocator, groundtruth = false
) where {T}
    linecount = 0
    index = 1
    _v, _dim = open(file) do io
        # First, read the dimensionality of the vectors in this file.
        # Make sure it is the same for all vectors.
        dim = read(io, Int32)

        # Since we want to support different allocators, we want to allocate all the memory
        # we're going to need up front.
        #
        # We do this by figuring out how many data points (aka lines) are in the dataset.
        # This is either based on the maximum number of requested lines, or the file size
        # (whichever is smaller).
        #
        # From there, we can work back how how many data points there will be and
        # preallocate the storage.
        linesize = sizeof(Int32) + dim * sizeof(vecs_read_type(T))
        num_lines, rem = divrem(filesize(file), linesize)
        @assert iszero(rem)
        if (maxlines !== nothing)
            num_lines = min(num_lines, maxlines)
        end
        meter = ProgressMeter.Progress(num_lines, 1, "Loading Dataset...")

        vector_len = div(num_lines * dim * sizeof(vecs_read_type(T)), sizeof(T))
        v = allocator(T, vector_len)

        buf = Vector{vecs_read_type(T)}(undef, dim)

        # Now, start parsing!
        while true
            read!(io, buf)
            # If we're reading in a groundtruth file, assume it's index zero and add 1 to
            # all indexes read.
            groundtruth && (buf .+= one(eltype(buf)))
            index += addto!(v, index, buf)

            linecount += 1
            ProgressMeter.next!(meter)
            ((linecount === maxlines) || eof(io)) && break

            # Read the next dimension. Make sure it's the same.
            nextdim = read(io, Int32)
            @assert dim == nextdim
        end
        ProgressMeter.finish!(meter)
        return v, dim
    end
    return vecs_reshape(T, _v, _dim)
end

"""
    save_vecs(io::Union{AbstractString, IO}, A::AbstractMatrix{T})

Save the matrix `A` to `io` in the `*vecs` format.
"""
function save_vecs(file::AbstractString, A::AbstractMatrix{T}) where {T}
    # Make the path if required
    dir = dirname(file)
    !ispath(dir) && mkpath(dir)

    # Drop down to an overloaded function that operates directly on a IO type object.
    open(file; write = true) do io
        save_vecs(io, A)
    end
end

function save_vecs(io::IO, A::AbstractMatrix{T}) where {T}
    meter = ProgressMeter.Progress(size(A, 2), 1)
    for col in eachcol(A)
        write(io, Int32(length(col)))
        write(io, col)
        ProgressMeter.next!(meter)
    end
end

