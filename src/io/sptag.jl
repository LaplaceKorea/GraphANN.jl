struct SPTAG <: AbstractIOFormat end

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
        # Optionaly print metadata.
        # Always need to follow with a tab, even if there is no metadata.
        metadata === nothing || (print(io, metadata[i]))
        print(io, '\t')
        # Dump delimiter-separated numbers.
        for j in eachindex(col)
            print(io, col[j])
            if j != lastindex(col)
                print(io, delimiter)
            end
        end
        # End of line.
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
            i != lastindex(col) && print(io, delimiter)
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
sentinel_value(::Type{T}) where {T <: Signed} = -one(T)
sentinel_value(x::T) where {T <: Integer} = sentinel_value(T)

issentinel(x) = (x == sentinel_value(x))
issentinel(x::TreeNode) = iszero(x.id)

maybeadjust(x::Integer, add = zero(x)) = issentinel(x) ? zero(x) : (x + add)

# Nodes get loaded with index-0 indices for the point ids.
# This is a helper function to convert everything to index-1
# However, the index ranges for the child nodes work fine if we take the very first node
# as the root.
#
# BUT, it's never that easy. We need to subtract 1 from the child-end size of things
# because C is crazy and doesn't include the last element of its ranges.
function adjust(x::TreeNode{T}) where {T}
    return TreeNode(
        # Bump the ID up from index-0 to index-1
        maybeadjust(x.id, one(T)),
        # C++ start indices are already valid for index-1.
        # Just turn "-1" bit patterns into "0".
        maybeadjust(x.childstart),
        # If this is a valid node, decrement the upper bound so the range is inclusive.
        maybeadjust(x.childend, -one(T)),
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

    # Number of TreeNode in the
    # We first read the root node which seems to be special.
    number_of_nodes = read(io, UInt32)
    root = adjust(read(io, TreeNode{UInt32}))
    nodes = Vector{TreeNode{UInt32}}(undef, number_of_nodes - 1)
    read!(io, nodes)
    nodes .= adjust.(nodes)

    # SPTAG applies a dummy-node at the end of the vector.
    # Here, we drop the dummy node so that the resulting tree passes our internal checks.
    resize!(nodes, length(nodes) - 1)

    if !eof(io)
        error("Something went wrong when loading BKTree. End of file not reached!")
    end

    # TODO: remove sentinel nodes that show up at the end of the file ...
    tree = Tree{UInt32}(root.childend, nodes)
    return (; number_of_trees, tree_start_indices, tree)
end

_decrement(x::UInt32, add = zero(x)) = iszero(x) ? sentinel_value(x) : (x - add)
function _decrement(x::TreeNode{UInt32})
    return TreeNode(
        # Convert from index-1 to index-0
        _decrement(x.id, one(UInt32)),
        # Julia indices are valid C++ indices because of how the data is structured in
        # SPTAG.
        #
        # Just turn "0" into all ones bit pattern.
        _decrement(x.childstart),

        # If this is not a child node, bump the last index for a valid C++ range.
        _decrement(x.childend, -one(UInt32)),
    )
end

function save(::SPTAG, io::IO, tree::Tree{UInt32})
    bytes_written = 0
    # Only saving 1 tree.
    bytes_written += write(io, Cuint(1))

    # Start indices for the tree encodings - only 1 tree, so only one entry.
    bytes_written += write(io, Cuint(0))

    # Number of nodes
    # Add 1 to include the end node and root node.
    bytes_written += write(io, Cuint(length(tree) + 2))

    # The root node is implicitly defined in our encoding.
    # Here, we explicitly instantiate it for serializing.
    bytes_written += write(io, Cuint(length(tree)))
    bytes_written += write(io, one(Cuint))
    bytes_written += write(io, Cuint(last(rootindices(tree)) + 1))

    # Write each node - converting from the Julia format to the C++ format>
    ProgressMeter.@showprogress 1 for node in tree
        bytes_written += write(io, _decrement(node))
    end

    # Finally, write a trailing sentinel node.
    for _ in 1:3
        bytes_written += write(io, sentinel_value(UInt32))
    end
    return bytes_written
end

#####
##### Graph Loading
#####

function load_graph(sptag::SPTAG, path::AbstractString; kw...)
    return open(path; read = true) do io
        load_graph(sptag, io; kw...)
    end
end

# Graphs in SPTAG land are stored as a dense 2D matrix where the adjacency lists are stored
# along the primary memory axis.
#
# The last portion of each adjacency list is padded by "-1".
# This function both determines how many entries are valid and sorts the valid entries
# in each adjacency list.
#
# It ALSO convertes from index-0 to index-1.
function getlength!(v::AbstractVector{T}, max_degree, sort::Bool) where {T <: Unsigned}
    i = findfirst(issentinel, v)
    num_neighbors = (i === nothing) ? T(max_degree) : T(i - 1)
    vw = view(v, 1:num_neighbors)
    # Sort and convert to index-1
    sort && sort!(vw; alg = QuickSort)
    vw .+= one(T)

    return num_neighbors
end

# Note: `sort = false` is for testing purposes only.
function load_graph(::SPTAG, io::IO; allocator = stdallocator, sort = true)
    num_vertices = read(io, UInt32)
    max_degree = read(io, UInt32)
    adj = allocator(UInt32, max_degree, num_vertices)
    read!(io, vec(adj))
    lengths = map(x -> getlength!(x, max_degree, sort), eachcol(adj))
    return UniDirectedGraph{UInt32}(FlatAdjacencyList{UInt32}(adj, lengths))
end

function save(::SPTAG, io::IO, graph::UniDirectedGraph{UInt32})
    num_vertices = LightGraphs.nv(graph)
    max_degree = maximum(LightGraphs.outdegree(graph))

    bytes_written = 0
    bytes_written += write(io, Cuint(num_vertices))
    bytes_written += write(io, Cuint(max_degree))
    buffer = UInt32[]
    ProgressMeter.@showprogress 1 for v in LightGraphs.vertices(graph)
        neighbors = LightGraphs.outneighbors(graph, v)
        resize!(buffer, length(neighbors))
        buffer .= neighbors .- one(eltype(neighbors))
        bytes_written += write(io, buffer)
        # Pad with all ones bits.
        for _ in 1:(max_degree - length(neighbors))
            bytes_written += write(io, typemax(Cuint))
        end
    end
    return bytes_written
end

#####
##### Extra Misc Files
#####

function generate_misc(::SPTAG, dir::AbstractString, data::AbstractVector{<:AbstractVector})
    !isdir(dir) && mkdir(dir)

    num_vertices = length(data)

    # metadata.bin
    touch(joinpath(dir, "metadata.bin"))

    # metdataIndex.bin
    open(joinpath(dir, "metadataIndex.bin"); write = true) do io
        # First entry is the number of vertices
        write(io, Cuint(num_vertices))

        # Two mysterious UInt32 zeros
        write(io, zero(Cuint))
        write(io, zero(Cuint))

        # One 64-bit zero integer per vertex.
        for _ in 1:num_vertices
            write(io, zero(UInt64))
        end
    end

    # deletes.bin
    open(joinpath(dir, "deletes.bin"); write = true) do io
        write(io, zero(UInt32))
        write(io, Cuint(num_vertices))
        write(io, one(UInt32))

        # 0xff for each vertex.
        for _ in 1:num_vertices
            write(io, typemax(UInt8))
        end
    end

    # vectors.bin
    open(joinpath(dir, "vectors.bin"); write = true) do io
        write(io, Cuint(num_vertices))
        write(io, Cuint(length(eltype(data))))
        write(io, data)
    end
    return nothing
end
