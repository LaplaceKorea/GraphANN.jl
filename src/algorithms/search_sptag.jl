struct TreeSearcher{I,D,V}
    worklist::DataStructures.BinaryMinHeap{Neighbor{TreeNode{I},D}}
    best::BoundedMaxHeap{Neighbor{I,D}}
    visited::V
end

function TreeSearcher(
    tree::Tree,
    data::AbstractVector{T},
    heapsize::Integer;
    visited = nothing
) where {T}
    D = _Base.costtype(T)
    worklist = DataStructures.BinaryMinHeap{Neighbor{TreeNode{UInt32}, D}}()
    best = BoundedMaxHeap{Neighbor{UInt32, D}}(heapsize)
    return TreeSearcher(worklist, best, visited)
end

function Base.empty!(algo::TreeSearcher)
    # Slight hijack. Technically, we don't own "valtree"
    empty!(algo.worklist.valtree)
    empty!(algo.best)
    tracksvisited(algo) && empty!(algo.visited)
    return nothing
end

function _Base.Neighbor(algo::TreeSearcher, id::T, distance::D) where {T,D}
    return Neighbor{T,D}(id, distance)
end

# TODO: I'm not so sure about these definitions ...
getnode(x::Neighbor{<:TreeNode}) = x.id
_Base.getid(x::Neighbor{<:TreeNode}) = getid(getnode(x))

isdone(x::TreeSearcher) = isempty(x.worklist)
pushcandidate!(x::TreeSearcher, candidate::Neighbor) = push!(x.worklist, candidate)
# Don't add a candidate if it's already been visited.
function maybe_pushcandidate!(x::TreeSearcher, candidate::Neighbor)
    if isvisited(x, candidate)
        println("Already seen: $candidate")
        return nothing
    end
    visited!(x, candidate)
    pushcandidate!(x, candidate)
    return  nothing
end
getcandidate!(x::TreeSearcher) = pop!(x.worklist)
pushbest!(x::TreeSearcher, candidate::Neighbor) = push!(x.best, candidate)

tracksvisited(::TreeSearcher{<:Any,<:Any,Nothing}) = false
tracksvisited(::TreeSearcher) = true

isvisited(x::TreeSearcher, node) = tracksvisited(x) && _isvisited(x, node)
_isvisited(x::TreeSearcher, node) = in(getid(node), x.visited)

visited!(x::TreeSearcher, node) = tracksvisited(x) && _visited!(x, node)
_visited!(x::TreeSearcher, node) = push!(x.visited, getid(node))

function _Base.search(
    algo::TreeSearcher,
    tree::Tree,
    data::AbstractVector{T},
    query::U,
    numleaves::Integer;
    metric = distance,
) where {T,U}
    empty!(algo)
    @unpack root, nodes = tree

    # Supply initial seeds.
    for i in childindices(root)
        child = nodes[i]
        maybe_pushcandidate!(
            algo,
            Neighbor(algo, child, metric(data[getid(child)], query))
        )
    end

    # Start processing!
    leaves_seen = 0
    while !isdone(algo)
        neighbor = getcandidate!(algo)
        node = getnode(neighbor)

        if isleaf(node)
            pushbest!(
                algo,
                Neighbor(algo, getid(neighbor), getdistance(neighbor))
            )
            leaves_seen += 1
            leaves_seen > numleaves && return break
        else
            pushbest!(algo, Neighbor(algo, getid(neighbor), getdistance(neighbor)))
            for i in childindices(getnode(neighbor))
                child = nodes[i]
                @unpack id = child
                iszero(id) && continue
                maybe_pushcandidate!(
                    algo,
                    Neighbor(algo, child, metric(data[id], query))
                )
            end
        end
    end
    return nothing
end
