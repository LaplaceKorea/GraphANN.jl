# Note - this should shadow the `TreeNode` type in the C++ code, which uses `Int32/UInt32`
# for the internal fields.
#
# In the C++ code, leaf nodes use Int32(-1) / typemax(UInt32) to encode the child nodes.
# Locally, we use "0" for the child fields of leaves.
struct TreeNode{T <: Integer}
    id::T
    childstart::T
    childend::T
end

TreeNode{T}() where {T <: Integer} = TreeNode{T}(zero(T), zero(T), zero(T))
function TreeNode{T}(id::Integer) where {T <: Integer}
    return TreeNode{T}(id, zero(T), zero(T))
end
TreeNode{T}(x::TreeNode{T}) where {T} = x

_Base.getid(x::TreeNode) = x.id
isleaf(x::TreeNode) = iszero(x.childstart)
isnull(x::TreeNode) = iszero(x.id)

childindices(x::TreeNode) = (x.childstart:x.childend)
Base.isless(a::T, b::T) where {T <: TreeNode} = isless(getid(a), getid(b))
Base.broadcastable(x::TreeNode) = (x,)

function Base.read(io::IO, ::Type{TreeNode{T}}) where {T}
    id = read(io, T)
    childstart = read(io, T)
    childend = read(io, T)
    return TreeNode{T}(id, childstart, childend)
end

function Base.write(io::IO, x::TreeNode)
    write(io, x.id)
    write(io, x.childstart)
    write(io, x.childend)
end

#####
##### Tree
#####

# The layout of this data structure is pretty similar to to the layout of the SPTAG C++ code.
# I'm not sure yet if that is the best layout, but we have to start somewhere.
mutable struct Tree{T <: Integer} <: AbstractTree
    rootend::Int
    nodes::Vector{TreeNode{T}}
end

Base.getindex(tree::Tree, i::Integer) = tree.nodes[i]

function Tree{T}(num_elements::Integer; allocator = stdallocator) where {T}
    nodes = allocator(TreeNode{T}, num_elements)
    nodes .= TreeNode{T}()
    return Tree{T}(0, nodes)
end

rootindices(tree::Tree) = Base.OneTo(tree.rootend)
Base.length(tree::Tree) = length(tree.nodes)
Base.show(io::IO, tree::Tree) = print(io, "Tree with $(length(tree)) nodes")

children(tree::Tree, i::Integer) = children(tree, tree.nodes[i])
children(tree::Tree, node::TreeNode) = view(tree.nodes, childindices(node))
roots(tree) = view(tree.nodes, rootindices(tree))

Base.iterate(tree::Tree, state...) = iterate(tree.nodes, state...)

#####
##### Utility Functions
#####

struct TreeError <: Exception
    msg::String
end

function onnodes(f::F, tree::Tree{T}) where {F,T}
    stack = TreeNode{T}[]
    append!(stack, roots(tree))
    while !isempty(stack)
        node = pop!(stack)
        f(node)
        isleaf(node) || append!(stack, children(tree, node))
    end
end

function ispacked(tree::Tree)
    indices_seen = falses(length(tree))
    # Mark the root nodes as visited.
    indices_seen[rootindices(tree)] .= true
    onnodes(tree) do node
        isleaf(node) && return nothing
        for i in childindices(node)
            indices_seen[i] && throw(TreeError("Already seen node at index $(i)!"))
            indices_seen[i] = true
        end
        return nothing
    end
    if count(indices_seen) != length(tree)
        throw(TreeError("Not all indexes in the tree vector were visited!"))
    end
    return true
end

function allids(tree::Tree)
    ids_seen = falses(length(tree))
    onnodes(tree) do node
        id = getid(node)
        (id <= 0 || id > length(tree)) && throw(TreeError("Node $id is out of bounds!"))
        ids_seen[id] && throw(TreeError("Already saw node $(id)!"))
        ids_seen[id] = true
    end
    num_ids_seen = count(ids_seen)
    if num_ids_seen != length(tree)
        throw("Missed seeing $(length(tree) - num_ids_seen) nodes!")
    end
    return true
end

function validate(tree)
    ispacked(tree)
    allids(tree)
    return true
end

#####
##### Tools to help build a tree.
#####

mutable struct TreeBuilder{T <: Integer}
    tree::Tree{T}
    last_valid_index::Int
    # Add a lock to facilitate building the tree with multiple threads.
    # Needs to be reentrant since `addnodes!` can potentially call `initnodes!`, but
    # `initnodes!` may be called on its own - leading to potentially two acquires.
    lock::ReentrantLock
end

function TreeBuilder{T}(num_nodes::Integer; allocator = stdallocator) where {T}
    return TreeBuilder(Tree{T}(num_nodes; allocator), 0, ReentrantLock())
end

function finish(builder::TreeBuilder)
    if builder.last_valid_index != length(builder.tree)
        throw(TreeError("Tree has not finished being built!"))
    end
    validate(builder.tree)
    return builder.tree
end

function partialfinish(builder::TreeBuilder)
    if builder.last_valid_index != length(builder.tree)
        throw(TreeError("Tree has not finished being built!"))
    end
    ispacked(builder.tree)
    return builder.tree
end

function initnodes!(builder::TreeBuilder{T}, itr) where {T}
    index = Base.@lock builder.lock begin
        @assert iszero(builder.last_valid_index)
        nodes = builder.tree.nodes
        index = 0
        for node in itr
            index += 1
            nodes[index] = TreeNode{T}(node)
        end

        # Update data structures.
        builder.tree.rootend = index
        builder.last_valid_index = index
        index
    end
    return 1:index
end

"""
    addnodes!(builder, parent::Integer, nodes)

Add contents of `nodes` as children of `parent` in the tree currently being built.
Return the indices where `nodes` were inserted as a range.

This function is safe to call with multiple threads.
"""
function addnodes!(builder::TreeBuilder{T}, parent::Integer, itr; force = false) where {T}
    range = Base.@lock builder.lock begin
        # Allow for common path in code using a builder.
        iszero(parent) && return initnodes!(builder, itr)

        @unpack tree, last_valid_index = builder
        @unpack nodes = tree

        # Make sure the parent node is not null and has no children currently assigned to it.
        parentnode = nodes[parent]
        isnull(parentnode) && throw(ArgumentError("Parent index $parent has not been assigned yet!"))
        isleaf(parentnode) || throw(ArgumentError("Parent index $parent already has children!"))

        count = 0
        start = last_valid_index + 1
        index = last_valid_index
        for node in itr
            index += 1
            nodes[index] = TreeNode{T}(node)
        end
        stop = index

        # Update the parent node to point to its children.
        nodes[parent] = TreeNode{T}(parentnode.id, start, stop)
        builder.last_valid_index = stop
        start:stop
    end
    return range
end

struct NodeOffset
    val::Int
end

function Base.:+(node::TreeNode{T}, offset::NodeOffset) where {T}
    if isleaf(node)
        return node
    else
        @unpack val = offset
        return TreeNode{T}(getid(node), node.childstart + val, node.childend + val)
    end
end

"""
    addtree!(builder::TreeBuilder, parent, subtree::Tree)

Insert `subtree` into `builder` under `parent`.
"""
function addtree!(builder::TreeBuilder{T}, parent::Integer, subtree::Tree) where {T}
    return Base.@lock builder.lock begin
        # Since we have a whole tree we're splicing in, we can essentially just copy and
        # paste into the next free region in the WIP tree.
        #
        # However, we do need to apply an appropriate offset for child indices.
        @unpack tree, last_valid_index = builder
        @unpack nodes = tree

        # Step 1 - Update the parent to the correct region.
        parentnode = nodes[parent]
        root_insert_start = last_valid_index + 1
        root_insert_end = root_insert_start + subtree.rootend - 1
        nodes[parent] = TreeNode{T}(getid(parentnode), root_insert_start, root_insert_end)

        # Step 2 - Insert subtree into tree by applying an offset to all the child ranges.
        offset = NodeOffset(last_valid_index)
        for node in subtree.nodes
            last_valid_index += 1
            nodes[last_valid_index] = node + offset
        end
        @pack! builder = last_valid_index
        offset.val
    end
end
