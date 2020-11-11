# Simple container for passing around graph parameters.
struct GraphParameters
    alpha::Float64
    max_degree::Int
    window_size::Int
end

#####
##### Pruner
#####

# This is an auxiliary data structure that aids in the pruning process.
# See tests for basic functionality.
struct Pruner{T}
    # Items to prune
    items::T
    # Boolean vector to determine if an index has been pruned.
    pruned::Vector{Bool}
end

Pruner{T}() where {T} = Pruner(T[], Bool[])

function Base.empty!(x::Pruner)
    empty!(x.items)
    empty!(x.pruned)
end

function initialize!(x::Pruner, itr)
    empty!(x)

    # Add all items from the input iterator to the local item list.
    for i in itr
        push!(x.items, i)
    end

    # Intially, mark all items as valid (i.e. not pruned)
    resize!(x.pruned, length(x))
    x.pruned .= false
    return nothing
end

# NB: Only call BEFORE performing any pruning
# This WILL NOT maintain the pruned state correctly.
Base.sort!(x::Pruner, args...; kw...) = sort!(x.items, args...; kw...)

Base.length(x::Pruner) = length(x.items)
Base.eltype(x::Pruner) = eltype(x.items)

# Since intermediate items may be pruned, we need to indicate that the size of this
# type cannot be known in advance.
Base.IteratorSize(::Type{<:Pruner}) = Base.SizeUnknown()

ispruned(x::Pruner, i) = @inbounds(x.pruned[i])
prune!(x::Pruner, i) = @inbounds(x.pruned[i] = true)

function Base.iterate(x::Pruner, i = 1)
    # Move forward until we find a non-removed entry
    while (i <= length(x) && ispruned(x, i))
        i += 1
    end
    return i > length(x) ? nothing : (x.items[i], (i + 1))
end

# Mark entries as pruned according to some function `F`.
function prune!(f::F, x::Pruner, start = 1) where {F}
    for i in start:length(x)
        @inbounds begin
            if !ispruned(x, i) && f(x.items[i])
                prune!(x, i)
            end
        end
    end
end

#####
##### Vemana Indexing
#####

"""
This function roughly implements the `RobustPrune` algorithm presented in the paper.
One difference is that instead of directly mutating the graph, we populate a `nextlist`
that will contain the next neighbors for this node.

This allows threads to work in parallel with periodic synchronous updates to the graph.

*Note*: I'm pretty sure this technique is how it's implemented in the C++ code.

* `nextlist` - Vector to be populated for the next neighbors of the current node.
* `candidates` - Candidate nodes to be the next neighbors
* `vertex` - The current vertex this is being applied to.
* `meta` - Combination of Graph and Dataset.
* `parameters` - Parameters governing graph construction.
"""
function neighbor_updates!(
    nextlist::AbstractVector,
    candidates,
    vertex,
    meta::MetaGraph,
    parameters::GraphParameters,
    pruner::Pruner,
)
    # Destructure some parameters
    @unpack graph, data = meta
    @unpack alpha, max_degree = parameters

    # Update the candidates set to contain the outneighbors of the current vertex
    # and remove the current vertex.
    union!(candidates, LightGraphs.outneighbors(graph, vertex))
    delete!(candidates, vertex)

    # Sort the candidates by ...
    # TODO: Remove this array allocation
    vertex_data = data[vertex]
    initialize!(
        pruner,
        (Neighbor(u, distance(vertex_data, @inbounds data[u])) for u in candidates)
    )
    sort!(pruner)

    # As a precaution, make sure the list of updates for this node is empty.
    empty!(nextlist)
    for i in pruner
        # TODO: Use the starting mechanism in `prune!` to reduce the number of
        # computations required.
        push!(nextlist, getid(i))
        f = x -> (alpha * distance(data[i], data[x]) <= getdistance(x))
        prune!(f, pruner)
        length(nextlist) >= max_degree && break
    end
end

# Commit all the pending updates to the graph.
# Then, cycle util no nodes violate the degree requirement
function commit!(
    meta::MetaGraph,
    parameters::GraphParameters,
    nextlists,
    pruner,
)
    @unpack graph, data = meta
    @unpack max_degree = parameters


    # Replace the out neighbors
    T = eltype(graph)
    degree_violations = Set{T}()
    for (u, neighbors) in pairs(nextlists)
        sort!(neighbors)
        sorted_copy!(graph, u, neighbors)

        # Now, add back edges, tracking which vertices have degree violations
        for v in neighbors
            n = LightGraphs.add_edge!(graph, v, u)
            n > max_degree && push!(degree_violations, v)
        end
    end

    # Clean up all degree violations
    nextlist = Vector{eltype(graph)}()
    for v in degree_violations
        # Copy over the out neighbors to a buffer so we can also add the source vertex.
        neighbor_updates!(
            nextlist,
            # Since we've already added the back edge, the source vertex that caused the
            # overflow is already in the adjacency list for `v`
            Set(LightGraphs.outneighbors(graph, v)),
            v,
            meta,
            parameters,
            pruner,
        )

        sort!(nextlist)
        sorted_copy!(graph, v, nextlist)
    end
    return nothing
end

basevertices(x::Dict) = keys(x)

"""
Generate the Index for a dataset
"""
function generate_index(
    data,
    parameters::GraphParameters;
    sync_every = 100
)
    @unpack alpha, max_degree, window_size = parameters

    # Generate a random `max_degree` regular graph.
    pre_graph = LightGraphs.random_regular_digraph(length(data), max_degree)

    # NOTE: This is a hack for now - need to write a generator for the UniDiGraph.
    graph = UniDirectedGraph(LightGraphs.nv(pre_graph))
    for e in LightGraphs.edges(pre_graph)
        LightGraphs.add_edge!(graph, e)
    end

    meta = MetaGraph(graph, data)

    # TODO: replace with shuffle
    greedy = GreedySearch(window_size)
    pruner = Pruner{Neighbor}()
    nextlists = Dict{Int,Vector{Int}}()
    synccount = 0
    for i in 1:length(data)
        # Perform a greedy search from this node.
        # The visited list will live inside the `greedy` object and will be extracted
        # using the `getvisited` function.
        search(greedy, meta, i, data[i])
        candidates = getvisited(greedy)

        # TODO: Make this ... actually good ...
        # Run the `RobustPrune` algorithm on the graph starting at this point.
        # Array `nextlist` will contain the updated neighbors for this vertex.
        # We delay actually implementing the updates to facilitate parallelization.
        nextlist = Int[]
        neighbor_updates!(
            nextlist,
            candidates,
            i,
            meta,
            parameters,
            pruner
        )
        nextlists[i] = nextlist
        synccount += 1
        if synccount == sync_every
            println("Sync Count = $i")
            commit!(meta, parameters, nextlists, pruner)
            synccount = 0
        end
    end
    return graph
end

