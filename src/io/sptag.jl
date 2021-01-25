struct SPTAG end

# Generate an SPTAG compatible dataset.
function generate_data_file(sptag::SPTAG, path::String, x...; kw...)
    return open(path; write = true) do io
        generate_data_file(sptag, io, x...; kw...)
    end
end

function generate_data_file(
    ::SPTAG,
    io::IO,
    data::AbstractMatrix;
    metadata = nothing,
    delimiter = '|'
)
    for (i, col) in enumerate(eachcol(data))
        # Optionaly print metadata
        metadata === nothing || (print(io, metadata[i]))
        print(io, '\t')
        for j in eachindex(col)
            print(io, col[j])
            if j != lastindex(col)
                print(io, delimiter)
            end
        end
        # Add newline
        print(io, '\n')
    end
    return nothing
end

function generate_groundtruth_file(sptag::SPTAG, path::String, x...; kw...)
    return open(path; write = true) do io
        generate_groundtruth_file(sptag, io, x...; kw...)
    end
end

function generate_groundtruth_file(::SPTAG, io::IO, data::AbstractMatrix, delimiter = ' ')
    for col in eachcol(data)
        for i in eachindex(col)
            print(io, col[i])
            i != lastindex(col) && print(io, ' ')
        end
        print(io, '\n')
    end
    return nothing
end

#####
##### Parse .ini files
#####

parseini(sptag::SPTAG, path::AbstractString) = open(io -> parseini(sptag, io), path)
function parseini(::SPTAG, io::IO)
    outer = Dict{String,Any}()
    current_key = ""
    current_dict = Dict{String,Any}()
    for ln in Iterators.filter(!isempty, eachline(io))
        # See if we're dealing with a new outer field.
        newouter = match(r"\[(.*)\]", ln)
        if newouter !== nothing
            current_key = only(newouter.captures)
            current_dict = Dict{String,Any}()
            outer[current_key] = current_dict
            continue
        end

        # Now, check if we're currently working an dictionary.
        isempty(current_key) && error("""
            Parse failure. Expected first entry to match [stuff]!
            """)

        key, value = split(ln, '=')

        # Prioritize parsing as integers, then as floats
        isint = tryparse(Int64, value; base = 10)
        isfloat = tryparse(Float64, value)
        if isint !== nothing
            current_dict[key] = isint
        elseif isfloat !== nothing
            current_dict[key] = isfloat
        else
            current_dict[key] = value
        end
    end
    return outer
end

#####
##### Tree Readers
#####

sentinel_value(::Type{T}) where {T <: Unsigned} = typemax(T)
sentinel_value(::Type{T}) where {T <: Signed} = -(one(T))
sentinel_value(x::T) where {T <: Integer} = sentinel_value(T)

maybeadjust(x::Signed) = (x == sentinel_value(x)) ? zero(x) : x
maybeadjust(x::Unsigned) = (x == sentinel_value(x)) ? zero(x) : x
maybeincrement(x::Signed) = (x == sentinel_value(x)) ? zero(x) : (x + one(x))
maybeincrement(x::Unsigned) = (x == sentinel_value(x)) ? zero(x) : (x + one(x))
issentinel(x::BKTNode) = iszero(x.id)

# Nodes get loaded with index-0 indices for the point ids.
# This is a helper function to convert everything to index-1
# However, the index ranges for the child nodes work fine if we take the very first node
# as the root.
function increment(x::BKTNode{T}) where {T}
    return BKTNode(
        maybeincrement(x.id),
        maybeadjust(x.childstart),
        maybeadjust(x.childend),
    )
end

function load_bktree(path::AbstractString, x...; kw...)
    return open(io -> load_bktree(io, x...; kw...), path)
end

function load_bktree(io::IO)
    # First 4 bytes give the number of trees stored in this file.
    # For now, only support loading a single tree.
    number_of_trees = read(io, UInt32)

    if number_of_trees > 1
        error("Support for loading multiple BKTrees is not yet implemented!")
    end

    # This array stores where each tree begins in the vector of nodes that we are about
    # to read.
    #
    # Because this is Julia, we need to make sure to add 1 to the indices.
    tree_start_indices = Vector{UInt32}(undef, number_of_trees)
    read!(io, tree_start_indices)
    tree_start_indices = Int.(tree_start_indices) .+ 1

    # Number of BKTNodes in the
    # We first read the root node which seems to be special.
    number_of_nodes = read(io, UInt32)
    root = read(io, BKTNode{Int32})
    nodes = Vector{BKTNode{Int32}}(undef, number_of_nodes - 1)
    read!(io, nodes)
    nodes .= increment.(nodes)

    if !eof(io)
        error("Something went wrong when loading BKTree. End of file not reached!")
    end

    # TODO: remove sentinel nodes that show up at the end of the file ...
    tree = BKTree(root, nodes)
    return (; number_of_trees, tree_start_indices, tree)
end

