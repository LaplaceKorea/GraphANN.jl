"""
    MetaGraph{G,D}

Grouping of a graph of type `G` and corresponding vertex data points of type `D`.
"""
struct MetaGraph{G,D}
    graph::G
    data::D
end

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
struct UniDirectedGraph{T <: Integer} <: LightGraphs.AbstractSimpleGraph{T}
    # Only track forward adjacency lists
    fadj::Vector{Vector{T}}

    # -- Inner constructor to help with ambiguities
    UniDirectedGraph{T}(adj::Vector{Vector{T}}) where {T} = new{T}(adj)
end

UniDirectedGraph{T}(n::Integer = 0) where {T} = UniDirectedGraph{T}([T[] for _ in 1:n])
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
# TODO: Maybe binary search?
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

function Base.eltype(
    ::Type{LightGraphs.SimpleGraphs.SimpleEdgeIter{UniDirectedGraph{T}}}
) where {T}
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
    verts = LightGraphs.vertices(g)
    # Check if vertices are in bounds
    (in(s, verts) && in(d, verts)) || return 0
    @inbounds list = fadj(g, s)
    index = searchsortedfirst(list, d)

    # Is the edge already in the graph?
    @inbounds (index <= length(list) && list[index] == d) && return length(list)
    insert!(list, index, d)
    return length(list)
end

function LightGraphs.add_edge!(g::UniDirectedGraph, e::LightGraphs.SimpleGraphs.SimpleEdge)
    return LightGraphs.add_edge!(g, e.src, e.dst)
end

function LightGraphs.add_vertex!(g::UniDirectedGraph{T}) where {T}
    push!(g.fadj, T[])
    return LightGraphs.nv(g)
end

# Remove ALL out neighbors of a vertex.
Base.empty!(g::UniDirectedGraph, v) = empty!(fadj(g, v))

# Copy over an adjacency list, destroying the current adjacency list in the process
function sorted_copy!(g::UniDirectedGraph, v, A::AbstractArray)
    list = fadj(g, v)
    resize!(list, length(A))
    copyto!(list, A)
    return nothing
end

# Fallback definition for an arbitrary iterable
function sorted_copy!(g::UniDirectedGraph, v, itr)
    list = fadj(g, v)
    empty!(list)
    for i in itr
        push!(list, i)
    end
    return nothing
end

#####
##### Generators
#####

function random_regular(::Type{T}, nv, ne) where {T <: Integer}
    # Allocate destination space
    adj = map(1:nv) do _
        list = T[]
        sizehint!(list, ne)
        return list
    end

    # Populate
    Threads.@threads for i in 1:nv
        _populate!(adj[i], one(T):T(nv), ne; exclude = i)
    end
    return UniDirectedGraph{T}(adj)
end

# Only call this if `ne << length(r)`
function _populate!(x::AbstractVector, r::AbstractRange, ne::Integer; exclude = ())
    empty!(x)
    while length(x) < ne
        i = rand(r)
        if !in(i, exclude) && !in(i, x)
            push!(x, i)
        end
    end
    sort!(x)
end
