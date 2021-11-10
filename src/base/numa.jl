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

"""
    onnode(f, node::Integer; [indexzero::Bool])

Run the function `f` on a thread running on numa node `node`.
Keyword `indexzero` indicates whether `node` should be interpreted as an index-zero
number or not.
"""
function onnode(f::F, node; indexzero = false) where {F}
    # Find a thread to run this on.
    node_adjusted = node + Int(indexzero)
    node_zero = node_adjusted - 1

    tid = findfirst(isequal(node_adjusted), NUMAMAP)
    local retval
    on_threads(ThreadPool(tid:tid)) do
        # Bind to local node
        ptr = @ccall libnuma.numa_allocate_nodemask()::Ptr{Cvoid}
        @ccall libnuma.numa_bitmask_setbit(ptr::Ptr{Cvoid}, node_zero::Cint)::Ptr{Cvoid}
        @ccall libnuma.numa_bind(ptr::Ptr{Cvoid})::Cvoid
        try
            # perform call
            retval = f()
        finally
            # set policy back to "preferred"
            @ccall libnuma.numa_set_preferred(node_zero::Cint)::Cvoid
            @ccall libnuma.numa_bitmask_free(ptr::Ptr{Cvoid})::Cvoid
        end
    end
    return retval
end

struct OnNode{F}
    f::F
    node::Int
end
(f::OnNode)(args...) = onnode(() -> f.f(args...), f.node)

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

function _alloc_wrap(x::Union{AbstractArray, Tuple}, len)
    if length(x) != len
        error("Expected $len allocators! Instead got $(length(x))")
    end
    return x
end
_alloc_wrap(x, len) = [x for _ in Base.OneTo(len)]

function NumaAware(
    f::F; nodes = Base.OneTo(NUM_NUMA_NODES[]), allocator = stdallocator
) where {F}
    allocators = _alloc_wrap(allocator, length(nodes))
    copies = map(eachindex(nodes, allocators)) do i
        return f(OnNode(allocators[i], nodes[i]))
    end
    return NumaAware(copies)
end

const MaybeNumaAware{T} = Union{T, NumaAware{<:T}}

"""
    @numalocal expr

Convert the function call `expr` into an anonymout function taking an allocator.
Note, the expression `expr` must contain the keyword `__allocator__` for the location
for which the allocator will be substituded.

An example is given below
```julia
julia> f = GraphANN.@numalocal GraphANN.sample_dataset(; allocator = __allocator__);

julia> x = GraphANN.NumaAware(f);

julia> typeof(x)
GraphANN._Base.NumaAware{Vector{StaticArrays.SVector{128, Float32}}}
```
"""
macro numalocal(expr...)
    if length(expr) > 2
        error("Macro @numalocal takes at most 2 arguments")
    elseif length(expr) == 2
        keywords = [expr[1]]
        fcall = expr[2]
    else
        keywords = Expr[]
        fcall = expr[1]
    end

    # Process keywords
    kwmap = Dict{Symbol,Any}(
        :allocator => stdallocator
    )
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
    allocators = kwmap[:allocator]
    return quote
        NumaAware($(esc(sym)) -> $(esc(fcall)); allocator = $(esc(allocators)))
    end
end

