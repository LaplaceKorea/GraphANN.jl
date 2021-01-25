# Note - this should shadow the `BKTNode` type in the C++ code, which uses `Int32/UInt32`
# for the internal fields.
#
# Apparently, leaf nodes use Int32(-1) / typemax(UInt32) to encode the child nodes.
struct BKTNode{T <: Integer}
    id::T
    childstart::T
    childend::T
end

_Base.getid(x::BKTNode) = x.id
isleaf(x::BKTNode) = iszero(x.childstart)

function Base.read(io::IO, ::Type{BKTNode{T}}) where {T}
    id = read(io, T)
    childstart = read(io, T)
    childend = read(io, T)
    return BKTNode{T}(id, childstart, childend)
end

#####
##### BKTree
#####

# The layout of this data structure is pretty similar to to the layout of the SPTAG C++ code.
# I'm not sure yet if that is the best layout, but we have to start somewhere.
struct BKTree{T <: BKTNode}
    root::T
    nodes::Vector{T}
end

Base.show(io::IO, tree::BKTree) = print(io, "BKTree with $(length(tree.nodes)) nodes")

struct BKTreeNeighbor{T,D}
    node::BKTNode{T}
    distance::D
end

getnode(x::BKTreeNeighbor) = x.node
_Base.getdistance(x::BKTreeNeighbor) = x.distance
Base.isless(x::T, y::T) where {T <: BKTreeNeighbor} = isless(x.distance, y.distance)

function search(
    tree::BKTree,
    data::AbstractVector{T},
    query::U,
    numleaves::Integer,
    numneighbors::Integer;
    metric = distance,
) where {T,U}
    @unpack root, nodes = tree
    cost_type = _Base.cost_type(T, U)
    heap = BoundedMaxHeap{Neighbor{cost_type}}(numneighbors)
    worklist = DataStructures.BinaryMinHeap{BKTreeNeighbor{Int32, cost_type}}()

    # Supply initial seeds.
    for i in root.childstart:root.childend
        child = nodes[i]
        @unpack id = child
        push!(worklist, BKTreeNeighbor(child, metric(data[id], query)))
    end

    # Start processing!
    leaves_seen = 0
    while !isempty(worklist)
        neighbor = pop!(worklist)
        node = getnode(neighbor)
        if isleaf(node)
            push!(heap, Neighbor(getid(node), getdistance(neighbor)))
            leaves_seen += 1
            @show leaves_seen
            leaves_seen > numleaves && return heap
        else
            push!(heap, Neighbor(getid(node), getdistance(neighbor)))
            for i in node.childstart:node.childend
                child = nodes[i]
                @unpack id = child
                iszero(id) && continue
                push!(worklist, BKTreeNeighbor(child, metric(data[id], query)))
            end
        end
    end
    return nothing
end
