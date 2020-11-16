# Simple container for passing around graph parameters.
Base.@kwdef struct GraphParameters
    alpha::Float64
    window_size::Int

    # Three parameters describing graph construction
    target_degree::Int
    prune_threshold_degree::Int
    prune_to_degree::Int
end

function change_threshold(x::GraphParameters, threshold = x.target_degree)
    return Setfield.@set x.prune_threshold_degree = threshold
end

onealpha(x::GraphParameters) = Setfield.@set x.alpha = 1.0

#####
##### Pruner
#####

# This is an auxiliary data structure that aids in the pruning process.
# See tests for basic functionality.
struct Pruner{T}
    # Items to prune
    items::Vector{T}
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
initialize!(x::Pruner, itr) = initialize!(identity, _ -> true, x, itr)

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

function Base.sizehint!(x::Pruner, sz)
    sizehint!(x.items, sz)
    sizehint!(x.pruned, sz)
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

struct NextListBuffer{T}
    # Keep around previously allocated buffers
    buffers::Vector{Vector{T}}
    nextlists::Dict{Int, Vector{T}}
    sizehint::Int
end

# Constructor
function NextListBuffer{T}(sizehint::Integer, num_buffers::Integer = 1) where {T}
    buffers = map(1:num_buffers) do _
        v = T[]
        sizehint!(v, sizehint)
        return v
    end

    nextlists = Dict{Int, Vector{T}}()
    sizehint!(nextlists, num_buffers)
    return NextListBuffer{T}(buffers, nextlists, sizehint)
end

# If we have a previously allocated vector, return that.
# Otherwise, allocate a new buffer.
function Base.get!(x::NextListBuffer{T}) where {T}
    !isempty(x.buffers) && return pop!(x.buffers)
    new = T[]
    sizehint!(new, x.sizehint)
    return new
end

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
    pruner::Pruner,
    alpha::AbstractFloat,
    target_degree::Integer
)
    # Destructure some parameters
    @unpack graph, data = meta

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
        length(nextlist) >= target_degree && break

        # Next step in the iteration interface
        next = iterate(pruner, state)
    end
end

function apply_nextlists!(graph, locks, tls::ThreadLocal; empty = false)
    # First step - copy over the next lists
    Threads.@threads for _ in allthreads()
        # Get local storage for this thread.
        # No need to lock because work partitioning is done such that the roote
        # nodes are unique for each thread.
        #
        # ... however, sneaky bugs can sometimes occur if this invariante,
        # so lock just to be on the safe side.
        #
        # Fortunately, the chance of collision is pretty low, so it shouldn't take very
        # long to get the locks.
        storage = tls[]
        for (u, neighbors) in pairs(storage.nextlists)
            Base.@lock locks[v] copyto!(graph, u, neighbors)
        end

        empty && empty!(storage.nextlists)
    end
end

function backedges!(graph, locks, tls, needs_pruning, prune_threshold)
    # Merge all edges to add together
    Threads.@threads for _ in allthreads()
        storage = tls[]
        for (u, neighbors) in pairs(storage.nextlists)
            for v in neighbors
                n = Base.@lock locks[v] LightGraphs.add_edge!(graph, v, u)
                n > prune_threshold && (needs_pruning[v] = true)
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
    needs_pruning::Vector{Bool},
    tls::ThreadLocal,
)
    @unpack graph, data = meta
    @unpack prune_threshold_degree, prune_to_degree, alpha = parameters

    # Step 1 - update all the next lists
    apply_nextlists!(graph, locks, tls)

    # Step 2 - Add back edges
    # If this pass is relaxed, then increase the degree at which we require reducing.
    backedges!(graph, locks, tls, needs_pruning, prune_threshold_degree)

    # Step 3 - process all over subscribed vertices
    Threads.@threads for v in eachindex(needs_pruning)
        needs_pruning[v] || continue

        storage = tls[]
        nextlist = get!(storage.nextlists)

        neighbor_updates!(
            nextlist,
            # Since we've already added the back edge, the source vertex that caused the
            # overflow is already in the adjacency list for `v`
            LightGraphs.outneighbors(graph, v),
            v,
            meta,
            storage.pruner,
            alpha,
            prune_to_degree,
        )
        storage.nextlists[v] = nextlist
        needs_pruning[v] = false
    end

    # Step 4 - update nextlists for all over subscribed vertices
    apply_nextlists!(graph, locks, tls; empty = true)
    return nothing
end

"""
Generate the Index for a dataset
"""
function generate_index(
    data,
    parameters::GraphParameters;
    batchsize = 1000,
    tls_monitor = _ -> nothing
)
    @unpack alpha, window_size, target_degree, prune_threshold_degree, prune_to_degree = parameters

    # Generate a random `max_degree` regular graph.
    # NOTE: This is a hack for now - need to write a generator for the UniDiGraph.
    # TODO: Also, default the graph to UInt32's to save on space.
    # Keep this as a parameter so we can promote to Int64's if needed.
    graph = random_regular(
        UInt32,
        length(data),
        target_degree;
        max_edges = prune_threshold_degree,
    )

    # Create a spin-lock for each vertex in the graph.
    # This will be used during the `commit!` function to synchronize access to the graph.
    locks = [Base.Threads.SpinLock() for _ in 1:LightGraphs.nv(graph)]
    needs_pruning = [false for _ in 1:LightGraphs.nv(graph)]

    meta = MetaGraph(graph, data)

    tls = ThreadLocal(;
        greedy = GreedySearch(window_size),
        pruner = Pruner{Neighbor}(),
        nextlists = NextListBuffer{eltype(graph)}(
            target_degree,
            2 * ceil(Int, batchsize / Threads.nthreads()),
        ),
    )

    # First iteration - set alpha = 1.0
    for i in 1:2
        _parameters = (i == 2) ? change_threshold(parameters) : onealpha(parameters)

        _generate_index(
            meta,
            _parameters,
            tls,
            locks,
            needs_pruning,
            batchsize;
            tls_monitor = tls_monitor,
        )
    end

    return meta
end

@noinline function _generate_index(
    meta::MetaGraph,
    parameters::GraphParameters,
    tls::ThreadLocal,
    locks::AbstractVector,
    needs_pruning::Vector{Bool},
    batchsize::Integer;
    # Optional monitors on thread local storage
    tls_monitor = _ -> nothing,
)
    @unpack graph, data = meta
    @unpack alpha, target_degree = parameters

    # TODO: replace with shuffle
    num_batches = ceil(Int, length(data) / batchsize)
    progress_meter = ProgressMeter.Progress(num_batches, 1, "Computing Index...")
    for batch_number in 1:num_batches
        start = ((batch_number - 1) * batchsize) + 1
        stop = min(batch_number * batchsize, length(data))
        r = start:stop

        # Thread across the batch
        itertime = @elapsed Threads.@threads for i in r
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
                storage.pruner,
                alpha,
                target_degree,
            )
            storage.nextlists[i] = nextlist
        end

        # Update the graph.
        # We also have the option of recording metrics about the state of the thread local
        # storage to help appropriately size preallocation.
        tls_monitor(tls)
        synctime = @elapsed commit!(
            meta,
            parameters,
            locks,
            needs_pruning,
            tls,
        )

        tls_monitor(tls)
        ProgressMeter.next!(
            progress_meter;
            showvalues = ((:iter_time, itertime), (:sync_time, synctime)),
        )
    end
    return graph
end

#####
##### Get the size of TLS structures
#####

struct TLSMonitor
    record::Dict{Symbol,Int}
    samplerate::Float64
end

function TLSMonitor(samplerate = 0.01)
    record = Dict(
        :pruner_length => 0,
        :nextlist_length => 0,
        :max_buffer_size => 0,
    )
    return TLSMonitor(record, samplerate)
end

function (monitor::TLSMonitor)(tls::ThreadLocal)
    # Only kick in rarely.
    rand() > monitor.samplerate && return nothing
    record = monitor.record

    @unpack pruner_length, nextlist_length, max_buffer_size = record

    for storage in getall(tls)
        pruner_length = max(pruner_length, length(storage.pruner))

        nextlists = storage.nextlists
        nextlist_length = max(
            nextlist_length,
            length(nextlists.buffers) + length(nextlists.nextlists)
        )

        a = safe_maximum(length, nextlists.buffers)
        b = safe_maximum(length, values(nextlists.nextlists))
        max_buffer_size = max(max_buffer_size, a, b)
    end

    @pack! record = pruner_length, nextlist_length, max_buffer_size
    return nothing
end

