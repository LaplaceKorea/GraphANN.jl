# Constants
const INDEX_BALANCE_FACTOR = 64

#####
##### Index Building
#####

"""
Parameters that control DiskANN style index building.

## Fields

* `alpha::Float64`: Controls how aggressive the pruning process is.
    Generally, a value of `1.2` is fine.
* `window_size::Int64`: Controls the size of the search history window used during the search
    phases of index construction. Higher values will yield higher quality indexes but a
    longer build time.
* `target_degree::Int64`: The maximum degree in the final graph.
* `prune_threshold_degree::Int64`: Degree at which a vertex is pruned. Setting this higher
    than `target_degree` helps ensure that all vertices don't require pruning at the same time.
    Generally, a value of `1.3 * target_degree` is good.
* `prune_to_degree::Int64`: When a vertex is pruned, it's maximal degree after pruning will be
    set to this amount. This is a somewhat legacy field and setting this equal to
    `target_degree` is fine.

## Constructor

This type is constructed using keyword arguments for all fields as shown below.
```jldoctest
julia> parameters = GraphANN.DiskANNIndexParameters(alpha = 1.2, window_size = 128, target_degree = 64, prune_threshold_degree = 84, prune_to_degree = 64)
GraphANN.Algorithms.DiskANNIndexParameters(1.2, 128, 64, 84, 64)
```
"""
Base.@kwdef struct DiskANNIndexParameters
    alpha::Float64
    window_size::Int64

    # Three parameters describing graph construction
    target_degree::Int64
    prune_threshold_degree::Int64
    prune_to_degree::Int64
end

function change_threshold(x::DiskANNIndexParameters, threshold = x.target_degree)
    return Setfield.@set x.prune_threshold_degree = threshold
end

onealpha(x::DiskANNIndexParameters) = Setfield.@set x.alpha = 1.0

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

# Fancy Printing
function Base.show(io::IO, x::Pruner{T}) where {T}
    print(io, typeof(x), "([")
    items = x.items
    for i in eachindex(items)
        color = ispruned(x, i) ? (:red) : (:green)
        printstyled(items[i]; color = color)
        i == lastindex(items) || print(io, ", ")
    end
    return print(io, "])")
end

function Base.empty!(x::Pruner)
    empty!(x.items)
    return empty!(x.pruned)
end

# Mark as `inline` to avoid an allocation caused by creating closures.
#
# NOTE: It's kind of gross to have the `by` and `filter` arguments as normal arguments
# instead of keyword arguments, but when they are keyword arguments, the function closures
# still allocate.
#
# This MAY get fixed in Julia 1.6, which will const-propagate keyword arguments.
@inline function initialize!(by::B, filter::F, x::Pruner, itr) where {B,F}
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
    return i > length(x) ? nothing : (@inbounds(x.items[i]), (i + 1))
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
    return sizehint!(x.pruned, sz)
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
    nextlists::Dict{Int,Vector{T}}
    sizehint::Int
end

# Constructor
function NextListBuffer{T}(sizehint::Integer, num_buffers::Integer = 1) where {T}
    buffers = map(1:num_buffers) do _
        v = T[]
        sizehint!(v, sizehint)
        return v
    end

    nextlists = Dict{Int,Vector{T}}()
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
    return empty!(x.nextlists)
end

# Actuall empty everything.
function purge!(x::NextListBuffer)
    empty!(x.buffers)
    return empty!(x.nextlists)
end

# Dispatch on behavior based on Sets vs Arrays.
# Use a closure to avoid code that doesn't necessarily need to run.
maybeunion!(f::F, candidates::AbstractSet) where {F} = union!(candidates, f())
maybeunion!(f::F, _) where {F} = nothing

# """
#     neighbor_updates!(args...)
#
# This function roughly implements the `RobustPrune` algorithm presented in the paper.
# One difference is that instead of directly mutating the graph, we populate a `nextlist`
# that will contain the next neighbors for this node.
#
# This allows threads to work in parallel with periodic synchronous updates to the graph.
#
# *Note*: I'm pretty sure this technique is how it's implemented in the C++ code.
#
# * `nextlist` - Vector to be populated for the next neighbors of the current node.
# * `candidates` - Candidate nodes to be the next neighbors
# * `vertex` - The current vertex this is being applied to.
# * `index` - Combination of Graph and Dataset.
# * `parameters` - Parameters governing graph construction.
# * `pruner` - Preallocated `Pruner` to help remove neighbors that fall outside the
#     distance threshold defined by `parameters.alpha`
# """
function neighbor_updates!(
    nextlist::AbstractVector,
    candidates,
    vertex,
    index::DiskANNIndex,
    pruner::Pruner,
    alpha::AbstractFloat,
    target_degree::Integer,
)
    # Destructure some parameters
    @unpack graph, data, metric = index

    # If the passed `candidates` is a Set, then append the current out neighbors
    # of the query vertex to the set.
    #
    # However, if `candidates` is an Array, than don't do anything because that array
    # likely comes from a graph and we do NOT want to mutate it.
    maybeunion!(() -> LightGraphs.outneighbors(graph, vertex), candidates)

    # Use lazy functions to efficientaly initialize the Pruner object
    vertex_data = data[vertex]
    initialize!(
        u -> Neighbor(index, u, evaluate(metric, vertex_data, pointer(data, u))),
        !isequal(vertex),
        pruner,
        candidates,
    )
    sort!(pruner; alg = Base.QuickSort, order = ordering(metric))

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
        function f(x)
            scaled = alpha * evaluate(metric, pointer(data, i), pointer(data, x))
            return Base.lt(ordering(metric), scaled, getdistance(x))
        end

        prune!(f, pruner; start = state)
        length(nextlist) >= target_degree && break

        # Next step in the iteration interface
        next = iterate(pruner, state)
    end
end

function apply_nextlists!(graph, locks, tls::ThreadLocal; empty = false)
    # First step - copy over the next lists
    on_threads(allthreads()) do
        # Get local storage for this thread.
        # No need to lock because work partitioning is done such that the roote
        # nodes are unique for each thread.
        #
        # ... however, sneaky bugs can sometimes occur if this invariant,
        # so lock just to be on the safe side.
        #
        # Fortunately, the chance of collision is pretty low, so it shouldn't take very
        # long to get the locks.
        storage = tls[]
        for (u, neighbors) in pairs(storage.nextlists)
            Base.@lock locks[u] copyto!(graph, u, neighbors)
        end
        empty && empty!(storage.nextlists)
    end
end

function add_backedges!(graph, locks, tls, needs_pruning, prune_threshold)
    # Merge all edges to add together
    on_threads(allthreads()) do
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

function prune!(
    index::DiskANNIndex,
    parameters::DiskANNIndexParameters,
    tls::ThreadLocal,
    needs_pruning::Vector{Bool};
    range = eachindex(needs_pruning),
)
    @unpack graph = index
    @unpack prune_to_degree, alpha = parameters

    dynamic_thread(range, INDEX_BALANCE_FACTOR) do v
        needs_pruning[v] || return nothing

        storage = tls[]
        nextlist = get!(storage.nextlists)

        neighbor_updates!(
            nextlist,
            # Since we've already added the back edge, the source vertex that caused the
            # overflow is already in the adjacency list for `v`
            LightGraphs.outneighbors(graph, v),
            v,
            index,
            storage.pruner,
            alpha,
            prune_to_degree,
        )
        storage.nextlists[v] = nextlist
        needs_pruning[v] = false
        return nothing
    end
end

# Commit all the pending updates to the graph.
# Then, cycle util no nodes violate the degree requirement
function commit!(
    index::DiskANNIndex,
    parameters::DiskANNIndexParameters,
    locks::AbstractVector,
    needs_pruning::Vector{Bool},
    tls::ThreadLocal,
)
    @unpack graph = index
    @unpack prune_threshold_degree = parameters

    # Step 1 - update all the next lists
    apply_nextlists!(graph, locks, tls)

    # Step 2 - Add back edges
    add_backedges!(graph, locks, tls, needs_pruning, prune_threshold_degree)

    # Step 3 - process all over subscribed vertices
    prune!(index, parameters, tls, needs_pruning)

    # Step 4 - update nextlists for all over subscribed vertices
    apply_nextlists!(graph, locks, tls; empty = true)
    return nothing
end

"""
    build(dataset, parameters::DiskANNIndexParameters; kw...) -> DiskANNIndex

Construct a DiskANN style relative neighbor graph for `dataset` according to the provided
[`DiskANNIndexParameters`](@ref).

# Keywords
* `graph_type`: The type of adjacency list to use during graph construction. Valid
    arguments are `DefaultAdjacencyList` or `FlatAdjacencyList` with a given integer type
    parameter (i.e. either `UInt32` if `dataset` is small enough, or `UInt64` otherwise).
    Default: `DefaultAdjacencyList{UInt32}`.
* `allocator`: Control the allocator used to allocate the graph. Only applies if
    `graph_type <: FlatAdjacencyList`. Default: `stdallocator`.
* `no_progress::Bool`: Set to `true` to disable displaying a progress bar. Default: `false`.
* `graph::Union{Nothing, LightGraphs.AbstractGraph}`: This argument allows passing in an
    existing graph rather than creating one from scratch. If `graph !== nothing`, then
    arguments `graph_type` and `allocator` are ignored. Default: `nothing`.
* `metric`: The metric to use during graph construction. Default: [`Euclidean`](@ref).
* `batchsize`: Graph construction happens over batches of indices to aid in parallelism.
    This parameter controls the size of those batches. A value of `1000 * Threads.nthreads()`
    or `2000 * Threads.nthreads()` is generally sufficient.
"""
function _Base.build(
    data,
    parameters::DiskANNIndexParameters;
    graph_type = DefaultAdjacencyList{UInt32},
    allocator = stdallocator,
    no_progress = false,
    graph::Union{Nothing,LightGraphs.AbstractGraph} = nothing,
    metric = Euclidean(),
    batchsize = 1000,
)
    @unpack window_size, target_degree, prune_threshold_degree = parameters

    # Generate a random `max_degree` regular graph.
    # TODO: Also, default the graph to UInt32's to save on space.
    # Keep this as a parameter so we can promote to Int64's if needed.
    if graph === nothing
        graph = random_regular(
            graph_type,
            length(data),
            target_degree;
            max_edges = prune_threshold_degree,
            allocator = allocator,
        )
    end

    # Create a spin-lock for each vertex in the graph.
    # This will be used during the `commit!` function to synchronize access to the graph.
    locks = [Base.Threads.SpinLock() for _ in 1:LightGraphs.nv(graph)]
    needs_pruning = [false for _ in 1:LightGraphs.nv(graph)]

    index = DiskANNIndex(graph, data, metric)
    tls = ThreadLocal(;
        greedy = DiskANNRunner(index, window_size),
        pruner = Pruner{Neighbor{eltype(graph),costtype(metric, data)}}(),
        nextlists = NextListBuffer{eltype(graph)}(
            target_degree, 2 * ceil(Int, batchsize / Threads.nthreads())
        ),
    )

    # First iteration - set alpha = 1.0
    # Second iteration - decrease pruning threshold to `target_degree`
    #for i in 1:2
    for i in 1:2
        _parameters = (i == 1) ? onealpha(parameters) : parameters
        _generate_index(
            index,
            _parameters,
            tls,
            locks,
            needs_pruning,
            batchsize;
            no_progress = no_progress,
        )
    end

    # Final cleanup - enforce degree constraint.
    # Batch this step to avoid over allocating "nextlists"
    _parameters = change_threshold(parameters)
    @unpack prune_threshold_degree = _parameters
    for range in batched(LightGraphs.vertices(graph), batchsize)
        Threads.@threads for v in range
            if length(LightGraphs.outneighbors(graph, v)) > prune_threshold_degree
                @inbounds needs_pruning[v] = true
            end
        end
        prune!(index, _parameters, tls, needs_pruning; range = range)
        apply_nextlists!(graph, locks, tls; empty = true)
    end

    return index
end

@noinline function _generate_index(
    index::DiskANNIndex,
    parameters::DiskANNIndexParameters,
    tls::ThreadLocal,
    locks::AbstractVector,
    needs_pruning::Vector{Bool},
    batchsize::Integer;
    no_progress = false,
)
    @unpack graph, data = index
    @unpack alpha, target_degree = parameters

    # TODO: Shuffle visit order.
    num_batches = cdiv(length(data), batchsize)
    progress_meter = ProgressMeter.Progress(num_batches, 1, "Computing Index...")
    for r in batched(eachindex(data), batchsize)
        # Use dynamic load balancing.
        @withtimer "Generating Nextlists" begin
            itertime = @elapsed dynamic_thread(r, INDEX_BALANCE_FACTOR) do vertex
                # Get thread local storage
                storage = tls[]

                # Perform a greedy search from this node.
                # The visited list will live inside the `greedy` object and will be extracted
                # using the `getvisited` function.
                point = pointer(data, vertex)
                search(storage.greedy, index, point, StartNode(vertex, point))
                candidates = getvisited(storage.greedy)

                # Run the `RobustPrune` algorithm on the graph starting at this point.
                # Array `nextlist` will contain the updated neighbors for this vertex.
                # We delay actually implementing the updates to facilitate parallelization.
                nextlist = get!(storage.nextlists)
                neighbor_updates!(
                    nextlist,
                    candidates,
                    vertex,
                    index,
                    storage.pruner,
                    alpha,
                    target_degree,
                )
                storage.nextlists[vertex] = nextlist
            end
        end

        # Update the graph.
        @withtimer "Updating Graph" begin
            synctime = @elapsed commit!(index, parameters, locks, needs_pruning, tls)
        end

        no_progress || ProgressMeter.next!(
            progress_meter;
            showvalues = ((:iter_time, itertime), (:sync_time, synctime)),
        )
    end
    return graph
end
