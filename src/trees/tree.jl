# Note - this should shadow the `TreeNode` type in the C++ code, which uses `Int32/UInt32`
# for the internal fields.
#
# Apparently, leaf nodes use Int32(-1) / typemax(UInt32) to encode the child nodes.
struct TreeNode{T <: Integer}
    id::T
    childstart::T
    childend::T
end

_Base.getid(x::TreeNode) = x.id
isleaf(x::TreeNode) = iszero(x.childstart)
childindices(x::TreeNode) = (x.childstart:x.childend)
Base.isless(a::T, b::T) where {T <: TreeNode} = isless(getid(a), getid(b))

function Base.read(io::IO, ::Type{TreeNode{T}}) where {T}
    id = read(io, T)
    childstart = read(io, T)
    childend = read(io, T)
    return TreeNode{T}(id, childstart, childend)
end

#####
##### Tree
#####
# The layout of this data structure is pretty similar to to the layout of the SPTAG C++ code.
# I'm not sure yet if that is the best layout, but we have to start somewhere.
struct Tree{T <: TreeNode}
    root::T
    nodes::Vector{T}
end

Base.show(io::IO, tree::Tree) = print(io, "Tree with $(length(tree.nodes)) nodes")

function validate(tree::Tree{T}) where {T}
    @unpack root, nodes = tree
    indices_seen = falses(length(nodes))
    dfs_stack = T[]

    # Add the root children first.
    for i in childindices(root)
        indices_seen[i] = true
        push!(dfs_stack, nodes[i])
    end

    while !isempty(dfs_stack)
        node = pop!(dfs_stack)
        isleaf(node) && continue
        for i in childindices(node)
            if indices_seen[i]
                error("Already seen node at index $(i)!")
            end
            indices_seen[i] = true
            push!(dfs_stack, nodes[i])
        end
    end
    return indices_seen
end

