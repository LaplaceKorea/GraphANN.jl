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

# Mark as `inline` to avoid an allocation caused by creating closures.
@inline function initialize!(by::B, filt::F, x::Pruner, itr) where {B, F}
    empty!(x)

    # Add all items from the input iterator to the local item list.
    for i in Iterators.filter(filt, itr)
        push!(x.items, by(i))
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

struct NextListBuffer{T}
    # Keep around previously allocated buffers
    buffers::Vector{Vector{T}}
    nextlists::Dict{Int, Vector{T}}
end

# Constructor
function NextListBuffer{T}() where {T}
    buffers = Vector{T}[]
    nextlists = Dict{Int, Vector{T}}()
    return NextListBuffer{T}(buffers, nextlists)
end

# If we have a previously allocated vector, return that.
# Otherwise, allocate a new buffer.
get!(x::NextListBuffer{T}) where {T} = isempty(x.buffers) ? T[] : pop!(x.buffers)

Base.getindex(x::NextListBuffer, i) = x.nextlists[i]
Base.setindex!(x::NextListBuffer{T}, v::Vector{T}, i) where {T} = x.nextlists[i] = v

Base.pairs(x::NextListBuffer) = pairs(x.nextlists)

# Remove previous mappings but save the allocated vectors for the next iteration.
function Base.empty!(x::NextListBuffer)
    for v in values(x.nextlists)
        empty!(v)
        push!(x.buffers, v)
    end
    empty!(x.nextlists)
end

# Actuall empty everything.
function deepempty!(x::NextListBuffer)
    empty!(x.buffers)
    empty!(x.nextlists)
end

maybeunion!(f::F, candidates::AbstractSet) where {F} = union!(candidates, f())
maybeunion!(f::F, candidates::AbstractArray) where {F} = nothing

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

    # If the passed `candidates` is a Set, then append the current out neighbors
    # of the query vertex to the set.
    #
    # However, if `candidates` is an Array, than don't do anything because that array
    # likely comes from a graph and we do NOT want to mutate it.
    maybeunion!(() -> LightGraphs.outneighbors(graph, vertex), candidates)

    vertex_data = data[vertex]
    initialize!(
        u -> Neighbor(u, distance(vertex_data, @inbounds data[u])),
        !isequal(vertex),
        pruner,
        candidates,
    )
    sort!(pruner; alg = Base.QuickSort)

    # As a precaution, make sure the list of updates for this node is empty.
    empty!(nextlist)
    for i in pruner
        # TODO: Use the starting mechanism in `prune!` to reduce the number of
        # computations required.
        push!(nextlist, getid(i))

        # Note: We're indexing `data` with `Neighbor` objects, but that's fine because
        # we've defined that behavior in `utils.jl`.
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
    misc_set::AbstractSet,
)
    @unpack graph, data = meta
    @unpack max_degree = parameters

    # Replace the out neighbors
    T = eltype(graph)

    degree_violations = misc_set
    empty!(degree_violations)
    for (u, neighbors) in pairs(nextlists)
        sort!(neighbors; alg = Base.QuickSort)
        sorted_copy!(graph, u, neighbors)

        # Now, add back edges, tracking which vertices have degree violations
        for v in neighbors
            n = LightGraphs.add_edge!(graph, v, u)
            n > max_degree && push!(degree_violations, v)
        end
    end
    empty!(nextlists)

    # Clean up all degree violations
    nextlist = get!(nextlists)
    for v in degree_violations
        # Copy over the out neighbors to a buffer so we can also add the source vertex.
        neighbor_updates!(
            nextlist,
            # Since we've already added the back edge, the source vertex that caused the
            # overflow is already in the adjacency list for `v`
            LightGraphs.outneighbors(graph, v),
            v,
            meta,
            parameters,
            pruner,
        )

        sort!(nextlist; alg = Base.QuickSort)
        sorted_copy!(graph, v, nextlist)
    end
    return nothing
end

"""
Generate the Index for a dataset
"""
function generate_index(
    data,
    parameters::GraphParameters;
    sync_every = 1000
)
    @unpack alpha, max_degree, window_size = parameters

    # Generate a random `max_degree` regular graph.
    # NOTE: This is a hack for now - need to write a generator for the UniDiGraph.
    pre_graph = LightGraphs.random_regular_digraph(length(data), max_degree)
    graph = UniDirectedGraph(LightGraphs.nv(pre_graph))
    for e in LightGraphs.edges(pre_graph)
        LightGraphs.add_edge!(graph, e)
    end

    meta = MetaGraph(graph, data)

    # TODO: replace with shuffle
    greedy = GreedySearch(window_size)
    pruner = Pruner{Neighbor}()
    nextlists = NextListBuffer{eltype(graph)}()
    synccount = 0
    misc_set = RobinSet{Int}()
    for i in 1:length(data)
        # Perform a greedy search from this node.
        # The visited list will live inside the `greedy` object and will be extracted
        # using the `getvisited` function.
        search(greedy, meta, i, data[i])
        candidates = getvisited(greedy)

        # Run the `RobustPrune` algorithm on the graph starting at this point.
        # Array `nextlist` will contain the updated neighbors for this vertex.
        # We delay actually implementing the updates to facilitate parallelization.
        nextlist = get!(nextlists)
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
            commit!(meta, parameters, nextlists, pruner, misc_set)

            # Reset data structures
            synccount = 0
            empty!(nextlists)
        end
    end
    return graph
end

