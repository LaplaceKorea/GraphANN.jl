const libnuma = "libnuma"
Libdl.dlopen(libnuma)

# Initialize map of Thread-ID to NUMA node.
# Will only be populated if `JULIA_EXCLUSIVE` is set.
const NUMAMAP = fill(1, Threads.nthreads())
const NUM_NUMA_NODES = Ref{Int}(1)

"""
    getnode() -> Int

Return the index-1 based NUMA node for the current thread.
"""
getnode() = @inbounds(NUMAMAP[Threads.threadid()])

#####
##### Get the numa nodes for an array
#####

pagesize() = 4096
function findnuma(v::DenseArray)
    # Round down pointers to a multiple of the page size
    base = Ptr{Nothing}(pagesize() * div(UInt(pointer(v)), pagesize()))
    pages = [base + i * pagesize() for i in 0:div(sizeof(v), pagesize())]
    statuses = fill(typemax(Cint), length(pages))
    retval = @ccall libnuma.numa_move_pages(
        0::Cint,
        length(pages)::Culong,
        pages::Ptr{Ptr{Cvoid}},
        Ptr{Cint}()::Ptr{Cint},
        statuses::Ptr{Cint},
        0::Cint
    )::Cint
    iszero(retval) || systemerror("Something went wrong", retval)
    return statuses
end

#####
##### Numa Aware Allocators
#####

struct NumaAllocator
    node::Int
end
numa_allocator(node::Int; indexzero = false) = NumaAllocator(node + Int(indexzero))

function (alloc::NumaAllocator)(::Type{T}, dims...) where {T}
    bytes = sizeof(T) * prod(dims)
    ptr = @ccall libnuma.numa_alloc_on_node(bytes::Csize_t, (alloc.node - 1)::Cint)::Ptr{T}
    array = Base.unsafe_wrap(Array, ptr, dims; own = false)
    finalizer(array) do
        @ccall libnuma.numa_free(ptr::Ptr{T}, bytes::Csize_t)::Cvoid
    end
end

_initialize!(x) = x
_initialize!(x::AbstractArray) = x .= one(eltype(x))
_initialize!(x::AbstractArray{SVector{N,T}}) where {N,T} = x .= (ones(SVector{N,T}),)

"""
    onnode(f, node::Integer; indexzero::Bool = false, strict = true, initialize = true)

Run the function `f` on a thread running on numa node `node`.
Keyword `indexzero` indicates whether `node` should be interpreted as an index-zero
number or not.
"""
function onnode(f::F, node; indexzero = false, strict = true, initialize = true) where {F}
    # Find a thread to run this on.
    node_adjusted = node + Int(indexzero)
    node_zero = node_adjusted - 1

    # Find a thread on the requested NUMA node
    tid = findfirst(isequal(node_adjusted), NUMAMAP)
    local retval
    on_threads(ThreadPool(tid:tid)) do
        # If we're running under `strict`, the we manually set the affinity of this current
        # thread temporarily while we invoke the allocator.
        #
        # We always try to fallback to the `preferred` mode otherwise.
        #
        # If not running under strict, then ... well ... don't do this.
        # This should not be called ever in performance sensitive code, so we can deal with the
        # small overhead of always allocating a `nodemask` even if we never use it.
        ptr = @ccall libnuma.numa_allocate_nodemask()::Ptr{Cvoid}
        if strict
            @ccall libnuma.numa_bitmask_setbit(ptr::Ptr{Cvoid}, node_zero::Cint)::Ptr{Cvoid}
            @ccall libnuma.numa_bind(ptr::Ptr{Cvoid})::Cvoid
        end
        try
            # perform call
            retval = f()
            initialize && _initialize!(retval)
        finally
            # set policy back to "preferred"
            # TODO: find what the existing policy was first instead of always defaulting
            # to preferred?
            @ccall libnuma.numa_set_preferred(node_zero::Cint)::Cvoid
            @ccall libnuma.numa_bitmask_free(ptr::Ptr{Cvoid})::Cvoid
        end
    end
    return retval
end

struct OnNode{F}
    f::F
    node::Int
    strict::Bool
end
OnNode(f, node::Integer) = OnNode(f, node, true)
(f::OnNode)(args...) = onnode(() -> f.f(args...), f.node; strict = f.strict)

#####
##### NumaAware
#####

struct NumaAware{T}
    copies::Vector{T}
    # Inner constructor to avoid method ambiguities with other single-argument constructors.
    NumaAware(copies::Vector{T}) where {T} = new{T}(copies)
end
Base.getindex(x::NumaAware) = x.copies[getnode()]
Base.getindex(x::NumaAware, i::Integer) = x.copies[i]
Base.length(x::NumaAware) = length(x.copies)

"""
    numalocal(x)

Return a version of `x` that is the closest in terms of NUMA distance.
"""
numalocal(x) = x
numalocal(x::NumaAware) = x[]

function _alloc_wrap(x::Union{AbstractArray, Tuple}, len)
    if length(x) != len
        error("Expected $len allocators! Instead got $(length(x))")
    end
    return x
end
_alloc_wrap(x, len) = [x for _ in Base.OneTo(len)]

function NumaAware(
    f::F; nodes = Base.OneTo(NUM_NUMA_NODES[]), allocator = stdallocator, strict = true
) where {F}
    allocators = _alloc_wrap(allocator, length(nodes))
    copies = map(eachindex(nodes, allocators)) do i
        return f(OnNode(allocators[i], nodes[i], strict))
    end
    return NumaAware(copies)
end

const MaybeNumaAware{T} = Union{T, NumaAware{<:T}}

maybe_escape(x::Union{Symbol,Expr}) = esc(x)
maybe_escape(x) = x

"""
    @numacopy [kw] expr

Construct multiple copies of the result of `expr` according to the number of NUMA nodes
on your system, each copy being allocated on its respective NUMA node.
The returned result will be a [`NumaAware`](@ref).

**Limitations**
1. The expression `expr` must contain somewhere within it the key word `__allocator__` where
   a NUMA bound allocator can be substituted. See the examples below.

2. You must be running with the environment variable `JULIA_EXCLUSIVE=1` for this to work
   correctly. Results without this setting may still work but will definitely not provide
   the performance benefit that NUMA-awareness can provide.

3. The expression `expr` must be safe to call multiple times.

Examples
--------
In the example below, we construct NUMA aware indices for the sample dataset and graph
packaged with the GraphANN code.
```julia-repl
julia> using GraphANN

# Allocate a dataset like normal
julia> data = GraphANN.sample_dataset(; allocator = GraphANN.stdallocator);

julia> typeof(data)

# Allocate a dataset on each NUMA node, using the default `stdallocator`
julia> data = GraphANN.@numacopy GraphANN.sample_dataset(; allocator = __allocator__);

julia> typeof(data)

julia> graph = GraphANN.@numacopy GraphANN.sample_graph(; allocator = __allocator__);

# We can construct an index and run queries like normal, even with the `NumaAware` wrappers
# around these data structures.
julia> index = GraphANN.DiskANNIndex(data, graph);
```

Keywords
--------
The behavior of this macro can be tuned by using keyword arguments taking the form
`key = value` and prefixed before the final expression. Explanations and examples are
provided below.

*   `allocator`: Supply either a single allocator or a `Vector`/`Tuple` of allocators equal
    equal in length to the number of NUMA nodes on your system. These allocators will be used
    for each inner function call. For example
```julia-repl
# Allocate all data structures using 1 GiB hugepages
julia> data = GraphANN.@numacopy allocator=GraphANN.huagepage_1gib_allocator begin
    GraphANN.sample_dataset(; allocator = __allocator__)
end;

# Alternatively, we can use [`pmallocators`](@ref) for the respective NUMA nodes
julia> allocators = (GraphANN.pmallocator("/mnt/pm0"), GraphANN.pmallocator("/mnt/pm1"));

julia> data = GraphANN.@numacopy allocator=allocators begin
    GraphANN.sample_dataset(; allocator = __allocator__)
end;
```
*   `strict::Bool`: On inner allocation calls, GraphANN will inform the Linux kernel that it
    should strictly obey local NUMA policies. If you wish to override this (for example,
    perhaps you have persistent memory modules mounted as system RAM and wish to use those
    even though they aren't necessarily the "default" NUMA node for a given CPU), then you
    can pass the `strict=false` keyword to avoid this check. Default: `true`.

*   `initialize::Bool`: If `true` - GraphANN will attempt to initialize allocated objects
    to ensure virtual memory pages get assigned from the correct NUMA node. Set this to
    `false` if this behavior is not desired. Default: `true`.
"""
macro numacopy(expr...)
    kwmap = Dict{Symbol,Any}(
        :allocator => stdallocator,
        :strict => true,
        :initialize => true,
    )
    nargs = length(kwmap) + 1

    if length(expr) > nargs
        error("Macro @numalocal takes at most $nargs arguments")
    end
    keywords = expr[1:end-1]
    fcall = expr[end]

    # Process keywords
    for kw in keywords
        @assert kw.head == :(=) && length(kw.args) == 2
        name = kw.args[1]
        value = kw.args[2]
        if haskey(kwmap, name)
            kwmap[name] = value
        else
            error("Unrecognized keyword $name")
        end
    end

    # Process function call into an anonymous
    sym = gensym(:__allocator)
    fcall = MacroTools.postwalk(fcall) do e
        if isa(e, Symbol) && e == :__allocator__
            return sym
        end
        return e
    end

    # Construct final expression
    allocators = maybe_escape(kwmap[:allocator])
    strict = maybe_escape(kwmap[:strict])
    return quote
        NumaAware(
            $(esc(sym)) -> $(esc(fcall));
            allocator = $allocators,
            strict = $strict,
        )
    end
end

