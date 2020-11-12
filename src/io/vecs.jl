# Preprocess a file with the expected element type.
ismaxed(x, ::Nothing) = false
ismaxed(x, y) = (x >= y)
ismatch(r::Regex, s) = (match(r, s) !== nothing)

# Auto detect format based on file extension.
# The return type of this function is not type-stable ... but that's okay, because it
# shouldn't be called in an performance critical code.
#
# Make sure any code operating on the return type of this function is located behind
# a function barrier.
#
# TODO: Allow creation of `points` types directly to automatically load data into AVX
# friendly chunks.
function load_vecs(file; kw...)
    # This will turn "/file/path/somefile.ivecs" into just ".ivecs".
    ext = last(splitext(file))

    # Is the extensions what we wanted it to be?
    if !ismatch(r"^\.(i|b|f)vecs", ext)
        throw(ArgumentError("Unknown file extension: \"$ext\"!"))
    end

    # What is the data type?
    datatype_char = ext[2]

    typemap = Dict(
        'b' => UInt8,
        'i' => UInt32,
        'f' => Float32,
    )

    return load_vecs(typemap[datatype_char], file; kw...)
end

function load_vecs(::Type{T}, file; maxlines = nothing) where {T}
    v = T[]
    sizehint!(v, div(filesize(file), sizeof(T)))

    linecount = 0
    dim = open(file) do io
        # First, read the dimensionality of the vectors in this file.
        # Make sure it is the same for all vectors.
        dim = read(io, Int32)
        buf = Vector{T}(undef, dim)

        # Now, start parsing!
        while true
            read!(io, buf)
            append!(v, buf)

            linecount += 1
            (ismaxed(linecount, maxlines) || eof(io)) && break

            # Read the next dimension. Make sure it's the same.
            nextdim = read(io, Int32)
            @assert dim == nextdim
        end
        return dim
    end
    return reshape(v, convert(Int, dim), :)
end

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

