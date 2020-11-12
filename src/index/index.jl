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
mutable struct Pruner{T}
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
#
# NOTE: It's kind of gross to have the `by` and `filter` arguments as normal arguments
# instead of keyword arguments, but when they are keyword arguments, the function closures
# still allocate.
#
# This MAY get fixed in Julia 1.6, which will const-propagate keyword arguments.
@inline function initialize!(
    by::B,
    filter::F,
    x::Pruner,
    itr;
) where {B, F}
    empty!(x)

    # Add all items from the input iterator to the local item list.
    for i in Iterators.filter(filter, itr)
        push!(x.items, by(i))
    end

    # Intially, mark all items as valid (i.e. not pruned)
    resize!(x.pruned, length(x))
    x.pruned .= false
    return nothing
end
initialize!(x::Pruner, itr) = initialize!(identity, x -> true, x, itr)

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
function prune!(f::F, x::Pruner; start = 1) where {F}
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

# Some implementation details
#
# 1. The algorithms for sorting are manually set to `QuickSort`.
#   This is because Julia defaults to `MergeSort`, which allocates temporary arrays.
#   On its own, this isn't too much of a problem, but allocations quickly become
#   troublesome when dealing with multi-threaded code.
#
#   `QuickSort`, on the other hand, does NOT allocate, so we use it.
#   TODO: `InsertionSort` might be something to look at if our vectors to sort are
#   pretty small ...
#
# 2. We try to make agressive resuse of existing allocations.
#   Types like the `NextListBuffer` below are abstractions to help make this optimization
#   a little easier to deal with in implementation code.

#####
##### NextListBuffer
#####

mutable struct NextListBuffer{T}
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
return!(x::NextListBuffer{T}, v::Vector{T}) where {T} = push!(x.buffers, v)

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
function purge!(x::NextListBuffer)
    empty!(x.buffers)
    empty!(x.nextlists)
end

# Dispatch on behavior based on Sets vs Arrays.
# Use a closure to avoid code that doesn't necessarily need to run.
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
* `pruner` - Preallocated `Pruner` to help remove neighbors that fall outside the
    distance threshold defined by `parameters.alpha`
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

    # Use lazy functions to efficientaly initialize the Pruner object
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

    # Manually lower the `for` loop to iteration so we can get the internal state of the
    # pruner iterator.
    #
    # This lets us start at the current location in the iterator to reduce the number
    # of comparisons.
    next = iterate(pruner)
    while next !== nothing
        (i, state) = next
        push!(nextlist, getid(i))

        # Note: We're indexing `data` with `Neighbor` objects, but that's fine because
        # we've defined that behavior in `utils.jl`.
        f = x -> (alpha * distance(data[i], data[x]) <= getdistance(x))
        prune!(f, pruner; start = state)
        length(nextlist) >= max_degree && break

        # Next step in the iteration interface
        next = iterate(pruner, state)
    end
end

#function apply_nextlists!(graph, locks, tls::ThreadLocal)
function apply_nextlists!(graph, locks, tls)
    # First step - copy over the next lists
    Threads.@threads for _ in allthreads()
        # Get local storage for this thread.
        storage = tls[]
        for (u, neighbors) in pairs(storage.nextlists)
            Base.@lock locks[u] sorted_copy!(graph, u, neighbors)
        end
    end
end

function backedges!(graph, locks, tls, max_degree)
    Threads.@threads for _ in allthreads()
        storage = tls[]
        degree_violations = storage.degree_violations
        empty!(degree_violations)

        for (u, neighbors) in pairs(storage.nextlists)
            for v in neighbors
                n = Base.@lock locks[v] LightGraphs.add_edge!(graph, v, u)
                n > max_degree && push!(degree_violations, v)
            end
        end
        empty!(storage.nextlists)
    end
end

# Commit all the pending updates to the graph.
# Then, cycle util no nodes violate the degree requirement
function commit!(
    meta::MetaGraph,
    parameters::GraphParameters,
    locks::AbstractVector,
    tls::ThreadLocal
)
    @unpack graph, data = meta
    @unpack max_degree = parameters

    # Step 1 - update all the next lists
    apply_nextlists!(graph, locks, tls)

    # Step 2 - Add back edges
    backedges!(graph, locks, tls, max_degree)

    # Step 3 - process all over subscribed vertices
    degree_violations = mapreduce(
        x -> x.degree_violations,
        union!,
        getall(tls);
        init = RobinSet{eltype(graph)}()
    ) |> collect

    Threads.@threads for v in degree_violations
        storage = tls[]
        nextlist = get!(storage.nextlists)

        neighbor_updates!(
            nextlist,
            # Since we've already added the back edge, the source vertex that caused the
            # overflow is already in the adjacency list for `v`
            LightGraphs.outneighbors(graph, v),
            v,
            meta,
            parameters,
            storage.pruner,
        )
        sort!(nextlist; alg = Base.QuickSort)
        storage.nextlists[v] = nextlist
    end

    # Step 4 - update nextlists for all over subscribed vertices
    apply_nextlists!(graph, locks, tls)
    return nothing
end

"""
Generate the Index for a dataset
"""
function generate_index(
    data,
    parameters::GraphParameters;
    batchsize = 1000,
)
    @unpack alpha, max_degree, window_size = parameters

    # Generate a random `max_degree` regular graph.
    # NOTE: This is a hack for now - need to write a generator for the UniDiGraph.
    # TODO: Also, default the graph to UInt32's to save on space.
    # Keep this as a parameter so we can promote to Int64's if needed.
    graph = random_regular(UInt32, length(data), max_degree)

    # Create a spin-lock for each vertex in the graph.
    # This will be used during the `commit!` function to synchronize access to the graph.
    locks = [Base.Threads.SpinLock() for _ in 1:LightGraphs.nv(graph)]
    meta = MetaGraph(graph, data)

    # Allocate thread local storage
    tls = ThreadLocal(;
        greedy = GreedySearch(window_size),
        pruner = Pruner{Neighbor}(),
        nextlists = NextListBuffer{eltype(graph)}(),
        degree_violations = RobinSet{eltype(graph)}(),
    )

    # First iteration - set alpha = 0
    initial_parameters = GraphParameters(1.0, max_degree, window_size)
    _generate_index(meta, initial_parameters, tls, locks, batchsize)

    # Second iteration, alpha = user defined
    _generate_index(meta, parameters, tls, locks, batchsize)
    return meta
end

@noinline function _generate_index(
    meta::MetaGraph,
    parameters::GraphParameters,
    tls::ThreadLocal,
    locks::AbstractVector,
    batchsize::Integer,
)
    @unpack graph, data = meta

    # TODO: replace with shuffle
    num_batches = ceil(Int, length(data) / batchsize)
    for batch_number in 1:num_batches
        start = ((batch_number - 1) * batchsize) + 1
        stop = min(batch_number * batchsize, length(data))
        r = start:stop

        # Thread across the batch
        nexttime = @elapsed Threads.@threads for i in r
            # Get thread local storage
            storage = tls[]

            # Perform a greedy search from this node.
            # The visited list will live inside the `greedy` object and will be extracted
            # using the `getvisited` function.
            search(storage.greedy, meta, i, data[i])
            candidates = getvisited(storage.greedy)

            # Run the `RobustPrune` algorithm on the graph starting at this point.
            # Array `nextlist` will contain the updated neighbors for this vertex.
            # We delay actually implementing the updates to facilitate parallelization.
            nextlist = get!(storage.nextlists)
            neighbor_updates!(
                nextlist,
                candidates,
                i,
                meta,
                parameters,
                storage.pruner,
            )

            sort!(nextlist; alg = Base.QuickSort)
            storage.nextlists[i] = nextlist
        end

        # Synchronize across threads.
        committime = @elapsed commit!(meta, parameters, locks, tls)
        println("Batch $batch_number of $num_batches. Inter Time: $nexttime. Sync Time: $committime")
    end

    return graph
end

