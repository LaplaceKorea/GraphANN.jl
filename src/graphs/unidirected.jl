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

"""
A directed graph that only stores the out-neighbors of each vertex.
Implements the [`AbststractGraph`](https://juliagraphs.org/LightGraphs.jl/latest/types/#AbstractGraph-Type)
interface.

In addition to the AbstractGraph interface, this type also implements
[`copyto!`](@ref Base.copyto!(::UniDirectedGraph, ::Integer, ::AbstractVector)) for fast
adjacency list updates.

## Adjacency List Choices

The `UniDirectedGraph` offers several choises for its adjacency list representation that
offer trade-offs in space efficiency, flexibility, and ease of allocation.
These choices are:

* [`DefaultAdjacencyList`](@ref): Maximum flexibility, good speed, no allocation choice.
* [`FlatAdjacencyList`](@ref): Moderate flexibility, high speed, easy allocation choice.
* [`DenseAdjacencyList`](@ref): Little fliexbility (no mutation), good speed, easy
    allocation choice.

## Simple Constructors

    UniDirectedGraph{T}([nv::Integer])

Construct an empty graph with vertex type `T` and `nv` vertices.

```jldoctest
julia> import LightGraphs

julia> graph = GraphANN.UniDirectedGraph{UInt32}(5)
{5, 0} directed simple UInt32 graph

julia> LightGraphs.add_edge!(graph, 1, 2)
1

julia> LightGraphs.add_vertex!(graph)
6

julia> LightGraphs.add_edge!(graph, 6, 4)
1

julia> collect(LightGraphs.edges(graph))
2-element Vector{LightGraphs.SimpleGraphs.SimpleEdge{UInt32}}:
 Edge 1 => 2
 Edge 6 => 4
```

## Flat Constructor

    UniDirectedGraph{T, FlatAdjacencyList}(nv, ne; [allocator])

Construct an empty graph with `nv` vertices backed by a [`FlatAdjacencyList`](@ref) that
can store a maximum of `ne` out neighbors per vertex.

```jldoctest
julia> import LightGraphs

julia> graph = GraphANN.UniDirectedGraph{UInt32, GraphANN.FlatAdjacencyList{UInt32}}(5, 2)
{5, 0} directed simple UInt32 graph

julia> LightGraphs.add_edge!(graph, 1, 2)
0x00000001

julia> LightGraphs.add_edge!(graph, 1, 3)
0x00000002

julia> LightGraphs.add_edge!(graph, 1, 4)
0x00000002

julia> graph # By construction, only a maximum of 2 neighbors per vertex
{5, 2} directed simple UInt32 graph

julia> collect(LightGraphs.edges(graph))
2-element Vector{LightGraphs.SimpleGraphs.SimpleEdge{UInt32}}:
 Edge 1 => 2
 Edge 1 => 3

julia> LightGraphs.add_vertex!(graph) # Cannot add vertices using this adjacency list
ERROR: Cannot yet push to a FlatAdjacencyList!
[...]
```

**Note**: Since this type only stores the out-neighbors, querying in-neighbors, while
supported, is quite slow.
"""
struct UniDirectedGraph{T<:Integer,A<:AbstractAdjacencyList{T}} <:
       LightGraphs.AbstractSimpleGraph{T}

    # Only track forward adjacency lists
    fadj::A

    # -- Inner constructor to help with ambiguities
    UniDirectedGraph{T}(adj::A) where {T,A<:AbstractAdjacencyList{T}} = new{T,A}(adj)
end

function UniDirectedGraph{T}(n::Integer = 0) where {T}
    fadj = DefaultAdjacencyList{T}([T[] for _ in 1:n])
    return UniDirectedGraph{T}(fadj)
end
UniDirectedGraph(n::Integer = 0) = UniDirectedGraph{Int}(n)

function UniDirectedGraph{T,FlatAdjacencyList{T}}(nv::Integer, ne::Integer; kw...) where {T}
    adj = FlatAdjacencyList{T}(nv, ne; kw...)
    return UniDirectedGraph{T}(adj)
end

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
function LightGraphs.ne(x::UniDirectedGraph{T,SuperFlatAdjacencyList{T}}) where {T}
    nv = LightGraphs.nv(x)
    iszero(nv) && return 0
    matrix = fadj(x).adj

    sums = ThreadLocal(0)
    dynamic_thread(Base.OneTo(nv), 2^16) do i
        sums[] += matrix[1,i]
    end
    return sum(getall(sums))
end

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
"""
    copyto!(g::UniDirectedGraph, v::Integer, A::AbstractVector)

Efficiently replace the out neighbors of `v` with the contents of `A`.
**Warning**: Vector `A` **may** be mutated as a side-effect of calling this function.
"""
Base.copyto!(g::UniDirectedGraph, v, A::AbstractVector) = copyto!(fadj(g), v, A)
_Base.unsafe_prefetch(g::UniDirectedGraph, v) = unsafe_prefetch(fadj(g), v)
