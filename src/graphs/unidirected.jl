#####
##### UniDirectedGraph
#####

# This is similar to a regular LightGraphs.SimpleDiGraph but with a few minor changes:
#
# 1. Only the forward adjacency list is maintained. The SimpleDiGraph records both forward
# and reverse adjacency lists. For our application, we only need one of those, reducing
# our memory footprint by half.
#
# 2. We add some convenience methods for destroying and quickly modifying adjacency lists
# that is helpful for some parts of the indexing algorithm.

# Is this an AbstractSimpleGraph or just an AbstractGraph?
struct UniDirectedGraph{
    T <: Integer,
    A <: AbstractAdjacencyList{T}
} <: LightGraphs.AbstractSimpleGraph{T}

    # Only track forward adjacency lists
    fadj::A

    # -- Inner constructor to help with ambiguities
    UniDirectedGraph{T}(adj::A) where {T, A <: AbstractAdjacencyList{T}} = new{T,A}(adj)
end

function UniDirectedGraph{T}(n::Integer = 0) where {T}
    fadj = DefaultAdjacencyList([T[] for _ in 1:n])
    return UniDirectedGraph{T}(fadj)
end
UniDirectedGraph(n::Integer = 0) = UniDirectedGraph{Int}(n)

LightGraphs.SimpleGraphs.fadj(x::UniDirectedGraph) = x.fadj
LightGraphs.SimpleGraphs.fadj(x::UniDirectedGraph, i::Integer) = x.fadj[i]

# We keep the invariant that vertices in the adjacency list must be sorted.
# Thus, we can use binary searches on the adjcancy lists to make things ever so slightly
# faster.
function _sorted_in(x, A)
    index = searchsortedfirst(A, x)
    return @inbounds (index <= length(A) && A[index] == x)
end

#####
##### LightGraph API implementation
#####

# Start with the easy ones
LightGraphs.nv(x::UniDirectedGraph) = length(fadj(x))
LightGraphs.ne(x::UniDirectedGraph) = iszero(LightGraphs.nv(x)) ? 0 : sum(length, fadj(x))
LightGraphs.outneighbors(x::UniDirectedGraph, v::Integer) = fadj(x, v)
LightGraphs.vertices(x::UniDirectedGraph) = Base.OneTo(LightGraphs.nv(x))

Base.eltype(::UniDirectedGraph{T}) where {T} = T

LightGraphs.edgetype(x::UniDirectedGraph) = LightGraphs.SimpleEdge{eltype(x)}
function LightGraphs.has_edge(x::UniDirectedGraph, s::Integer, d::Integer)
    return _sorted_in(d, fadj(x, s))
end

# Julia will specialize `in` for the Base.OneTo range returned by `vertices`.
LightGraphs.has_vertex(x::UniDirectedGraph, v::Integer) = in(v, LightGraphs.vertices(x))

LightGraphs.is_directed(::UniDirectedGraph) = true
LightGraphs.is_directed(::Type{<:UniDirectedGraph}) = true
Base.zero(::Type{UniDirectedGraph{T}}) where {T} = UniDirectedGraph{T}()

# Harder methods
function LightGraphs.inneighbors(x::UniDirectedGraph{T}, v::Integer) where {T}
    # Since we don't explicitly track inneighbors, we have to perform a linear
    # scan on the forward adjacency list.
    #
    # Here, we return a lazy iterable that searches for if the vertex in question is an
    # some other vertex's adjacencyh list.
    return Iterators.filter(i -> _sorted_in(v, fadj(x, i)), LightGraphs.vertices(x))
end

# This basically works because we implement `LightGraphs.SimpleGraphs.fadj`.
LightGraphs.edges(x::UniDirectedGraph) = LightGraphs.SimpleGraphs.SimpleEdgeIter(x)

const SimpleEdgeIter = LightGraphs.SimpleGraphs.SimpleEdgeIter
function Base.eltype(::Type{<:SimpleEdgeIter{<:UniDirectedGraph{T}}}) where {T}
    return LightGraphs.SimpleDiGraphEdge{T}
end

# Methods for adding edges
# Much of this is based on the LightGraph's implementation
#
# One difference here is that instead of returning `true` or `false`, we return the number
# of out neighbors of the source.
#
# During index creation, this lets us easily track which vertices exceed the maximum degree
# requirement.
function LightGraphs.add_edge!(g::UniDirectedGraph, s, d)
    adj = fadj(g)

    # If this adjacency list is already full, abort early.
    caninsert(adj, s) || return length(adj, s)

    # Check if vertices are in bounds
    verts = LightGraphs.vertices(g)
    (in(s, verts) && in(d, verts)) || return 0
    @inbounds list = adj[s]
    index = searchsortedfirst(list, d)

    # Is the edge already in the graph?
    @inbounds (index <= length(list) && list[index] == d) && return length(list)
    unsafe_insert!(adj, s, index, d)

    return length(adj, s)
end

function LightGraphs.add_edge!(g::UniDirectedGraph, e::LightGraphs.SimpleGraphs.SimpleEdge)
    return LightGraphs.add_edge!(g, e.src, e.dst)
end

function LightGraphs.add_vertex!(g::UniDirectedGraph{T}) where {T}
    push!(g.fadj, T[])
    return LightGraphs.nv(g)
end

# Remove ALL out neighbors of a vertex.
Base.empty!(g::UniDirectedGraph, v) = empty!(fadj(g), v)

# Copy over an adjacency list, destroying the current adjacency list in the process
# NOTE: We don't expect `A` to be sorted.
# In fact, it should probably NOT be sorted in order to ensure that the nearest-distance
# neighbors actually occur in the resulting adjacency list.
Base.copyto!(g::UniDirectedGraph, v, A::AbstractArray) = copyto!(fadj(g), v, A)

#####
##### Generators
#####

# function random_regular(
#     ::Type{T},
#     nv,
#     ne;
#     max_edges = ne,
#     slack = 1.05
# ) where {T <: Integer}
#
#     # Allocate destination space
#     adj = zeros(T, ceil(Int, slack * max_edges), nv)
#
#     _populate!(adj, ne)
#     lengths = fill(T(ne), nv)
#
#     return UniDirectedGraph{T}(FlatAdjacencyList{T}(adj, lengths))
# end

function random_regular(
    ::Type{T},
    nv,
    ne;
    max_edges = ne,
    slack = 1.05,
) where {T <: Integer}

    adj = Vector{Vector{T}}(undef, nv)
    _populate!(adj, ne, ceil(Int, slack * max_edges))
    @assert all(i -> isassigned(adj, i), 1:nv)

    return UniDirectedGraph{T}(DefaultAdjacencyList{T}(adj))
end

function _populate!(A::Matrix{T}, ne) where {T}
    tls = ThreadLocal(T[])
    nv = size(A, 2)

    Threads.@threads for col in 1:nv
        storage = tls[]
        empty!(storage)

        while length(storage) < ne
            i = rand(1:nv)
            if (i != col) && !in(i, storage)
                push!(storage, i)
            end
        end
        sort!(storage)
        @views A[1:ne, col] .= storage
    end
end

function _populate!(A::AbstractVector{<:AbstractVector{T}}, ne, max_edges) where {T}
    nv = length(A)

    Threads.@threads for col in 1:nv
        list = T[]
        sizehint!(list, max_edges)
        while length(list) < ne
            i = rand(1:nv)
            if (i != col) && !in(i, list)
                push!(list, i)
            end
        end
        sort!(list)
        A[col] = list
    end
    return A
end


# Only call this if `ne << length(r)`
#function _populate!(x::AbstractVector, r::AbstractRange, ne::Integer; exclude = ())
#    empty!(x)
#    while length(x) < ne
#        i = rand(r)
#        if !in(i, exclude) && !in(i, x)
#            push!(x, i)
#        end
#    end
#    sort!(x)
#end
