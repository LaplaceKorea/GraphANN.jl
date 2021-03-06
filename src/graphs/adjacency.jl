#####
##### AbstractAdjacencyList
#####

# Forward adjacency lists hold the outgoing neighbors for each vertex.
# The default implementation is to use a vector of vectors where:
# - The outer vector has one entry per vectex
# - The inner vectors store neighbors in ascending order (required by LightGraphs)
#
# This allows the ultimate flexibility for holding neighbor lists but it has a sneaky
# downside. When adding elements to the front or back of a vector, the memory backing
# the vector may need to be reallocated. Once the number of vertices becomes significantly
# large (on the order of 100's of millions), the time spend reallocating becomes PAINFUL.
#
# Thus, we abstract the behavior of the adjacency list and the graph types build on top
# of them.
#
# This allows an alternative implementation where adjacency lists are stored as one giant
# 2D matrix, with adjacency information living on the columns.
#
# Now, the following become restricted:
#
# - The maximum out degree of each vertex is now set.
# - It's really hard to add additional vertices.
#
# The latter COULD be solved by chunking these 2D arrays though ...
# Anyways, this pre-allocates all memory needed and avoids the potential reallocation
# penalty, at the cost of flexibility.

"""
Supertype for types implementing the `AbstractAdjacencyList{T}` api listed below.

* `getindex(x::AbstractAdjacencyList, v::Integer)`: Return the adjacency list for vertex `v`.
    The result should be an `AbstractVector{T}`.
* `caninsert(x::AbstractAdjacencyList, v::Integer)`: Return `true` if an aditional edge can
    be inserted for vertex `v`.
* `length(x::AbstractAdjacencyList)`: Return the number of vertices stored by `x`.
* `length(x::AbstractAdjacencyList, i::Integer)`: Return the number of outneighbors for
    vertex `i`.
* `copyto!(x::AbstractAdjacencyList, v, A::AbstractVector): Replace the adjacency list for
    `v` with the contents of `A`. **Note**: `A` does not need to be in any particular order,
    but **may** be mutated as part of this operation.

In addition, implementors of the `AbstractAdjacencyList` api must implement the iteration
interface to iterate over the adjacencylist of each vertex in order.
"""
abstract type AbstractAdjacencyList{T<:Integer} end
_Base.unsafe_prefetch(fadj::AbstractAdjacencyList, i) = unsafe_prefetch(fadj[i])

#####
##### Default implementation
#####

"""
Simple reference implementation of an adjacency list.
Implemented as a vector of vectors.
"""
struct DefaultAdjacencyList{T} <: AbstractAdjacencyList{T}
    fadj::Vector{Vector{T}}
end

DefaultAdjacencyList{T}() where {T} = DefaultAdjacencyList{T}(Vector{T}[])

Base.push!(x::DefaultAdjacencyList{T}, v::Vector{T}) where {T} = push!(x.fadj, v)
Base.getindex(x::DefaultAdjacencyList, i) = x.fadj[i]
caninsert(x::DefaultAdjacencyList, i) = true
function unsafe_insert!(x::DefaultAdjacencyList, v, index, value)
    # Do not use `Base.insert!` because this can shift either at the front or the back.
    # To keep vectors from walking too much, we always want to shift back.
    list = x[v]
    if index > length(list)
        push!(list, value)
        return nothing
    end

    # Move everything back
    resize!(list, length(list) + 1)
    num_to_move = length(list) - index

    unsafe_copyto!(list, index + 1, list, index, num_to_move)
    @inbounds list[index] = value
    return nothing
end

Base.length(x::DefaultAdjacencyList) = length(x.fadj)
Base.length(x::DefaultAdjacencyList, i) = length(x[i])
Base.empty!(x::DefaultAdjacencyList, i) = empty!(x[i])

Base.iterate(x::DefaultAdjacencyList, s...) = iterate(x.fadj, s...)

function Base.copyto!(x::DefaultAdjacencyList, v, A::AbstractVector)
    list = x[v]
    resize!(list, length(A))
    copyto!(list, A)
    return sort!(list; alg = Base.QuickSort)
end

#####
##### Flat Implementation
#####

abstract type AbstractFlatAdjacencyList{T} <: AbstractAdjacencyList{T} end

Base.push!(x::AbstractFlatAdjacencyList, v) = error("Cannot yet push to a FlatAdjacencyList!")

function Base.iterate(x::AbstractFlatAdjacencyList, s = 1)
    s > length(x) && return nothing
    return @inbounds (x[s], s + 1)
end

"""
Adjacency list implementation that allocates adjacency lists as a single 2D array.
Individual neighbor lists are found in the columns of the matrix, sorted from smallest to
largest with the lenght of the list stored in a `lengths` vector.

One advantage of this representation is that the entire list can be efficiently allocated
to persistent memory.

**Note**: Trying to additional neighbors to a vertex beyond the length of the columns will
silently become a no-op.
"""
struct FlatAdjacencyList{T} <: AbstractFlatAdjacencyList{T}
    adj::Matrix{T}
    # Store how many neighbors each vertex actually has.
    lengths::Vector{T}

    # -- Inner constructor to avoid ambiguity
    function FlatAdjacencyList{T}(adj::Matrix{T}, lengths::Vector{T}) where {T}
        return new{T}(adj, lengths)
    end
end

function FlatAdjacencyList{T}(
    nv::Integer, max_degree::Integer; allocator = stdallocator
) where {T}
    adj = allocator(T, max_degree, nv)
    dynamic_thread(eachindex(adj), 8192) do i
        @inbounds adj[i] = zero(T)
    end
    lengths = zeros(T, nv)
    return FlatAdjacencyList{T}(adj, lengths)
end

_max_degree(x::FlatAdjacencyList) = size(x.adj, 1)
Base.length(x::FlatAdjacencyList) = size(x.adj, 2)
Base.length(x::FlatAdjacencyList, i) = x.lengths[i]
Base.empty!(x::FlatAdjacencyList, i) = (x.lengths[i] = 0)

# Use `unsafe_view` because `lenght` will bounds check for us.
Base.getindex(x::FlatAdjacencyList, i) = Base.unsafe_view(x.adj, 1:length(x, i), Int(i))

# Can insert as long as the row is not completely full.
caninsert(x::FlatAdjacencyList, i) = (length(x, i) < _max_degree(x))

# This is marked unsafe, so assume bounds checking has already happened.
# Note: This is why we define `caninsert`
function unsafe_insert!(x::FlatAdjacencyList, v, index, value)
    @inbounds current_length = x.lengths[v]
    vw = Base.unsafe_view(x.adj, 1:(current_length + 1), v)

    num_to_move = current_length - index + 1
    if !iszero(num_to_move)
        src = pointer(vw, index)
        dst = pointer(vw, index + 1)
        unsafe_copyto!(dst, src, num_to_move)
    end
    @inbounds vw[index] = value
    @inbounds x.lengths[v] += 1
    return nothing
end

function Base.iterate(x::FlatAdjacencyList, s = 1)
    s > length(x) && return nothing
    return @inbounds (x[s], s + 1)
end

function Base.copyto!(x::FlatAdjacencyList, v, A::AbstractVector)
    # Resize - make sure we don't copy too many things
    md = _max_degree(x)
    len = min(length(A), _max_degree(x))
    x.lengths[v] = len

    sort!(A; alg = Base.QuickSort)

    # Start index for `dst` pointer: Compute linear offset based on the length of each
    # column.
    dst = pointer(x.adj, (md * (v - 1)) + 1)
    src = pointer(A)
    unsafe_copyto!(dst, src, len)
    return nothing
end

#####
##### SuperFlatAdjacencyList
#####

# Like a FlatAdjacencyList, but the number of neighbors is stored inline
struct SuperFlatAdjacencyList{T} <: AbstractFlatAdjacencyList{T}
    adj::Matrix{T}
    SuperFlatAdjacencyList{T}(adj::Matrix{T}) where {T} = new{T}(adj)
end

function SuperFlatAdjacencyList{T}(
    nv::Integer, max_degree::Integer; allocator = stdallocator
) where {T}
    # Allocate one extra slot in each column to store the length as the first entry.
    adj = allocator(T, max_degree + 1, nv)
    dynamic_thread(eachindex(adj), 8192) do i
        @inbounds adj[i] = zero(T)
    end
    return SuperFlatAdjacencyList{T}(adj)
end

function _Base.unsafe_prefetch(fadj::SuperFlatAdjacencyList, i)
    sz = size(fadj.adj, 1)
    offset = sz * i + 1
    unsafe_prefetch(pointer(fadj.adj, offset), sz)
end

_max_degree(x::SuperFlatAdjacencyList) = size(x.adj, 1) - 1
Base.length(x::SuperFlatAdjacencyList) = size(x.adj, 2)
Base.length(x::SuperFlatAdjacencyList, i) = x.adj[1, i]
setlength!(x::SuperFlatAdjacencyList, v, i) = x.adj[1, i] = v
Base.empty!(x::SuperFlatAdjacencyList, i) = setlength!(x, 0, i)

# Use `unsafe_view` because `lenght` will bounds check for us.
Base.getindex(x::SuperFlatAdjacencyList, i) = Base.unsafe_view(x.adj, 2:(1+length(x, i)), Int(i))

# Can insert as long as the row is not completely full.
caninsert(x::SuperFlatAdjacencyList, i) = (length(x, i) < _max_degree(x))

# This is marked unsafe, so assume bounds checking has already happened.
# Note: This is why we define `caninsert`
function unsafe_insert!(x::SuperFlatAdjacencyList, v, index, value)
    @inbounds current_length = length(x, v)
    vw = Base.unsafe_view(x.adj, 2:(current_length + 2), v)

    num_to_move = current_length - index + 1
    if !iszero(num_to_move)
        src = pointer(vw, index)
        dst = pointer(vw, index + 1)
        unsafe_copyto!(dst, src, num_to_move)
    end
    @inbounds vw[index] = value
    @inbounds setlength!(x, current_length + 1, v)
    return nothing
end

function Base.copyto!(x::SuperFlatAdjacencyList, v, A::AbstractVector)
    # Resize - make sure we don't copy too many things
    md = _max_degree(x)
    len = min(length(A), md)
    setlength!(x, len, v)
    sort!(A; alg = Base.QuickSort)

    # Start index for `dst` pointer: Compute linear offset based on the length of each
    # column.
    dst = pointer(x.adj, ((md + 1) * (v - 1)) + 2)
    src = pointer(A)
    unsafe_copyto!(dst, src, len)
    return nothing
end

#####
##### DenseAdjacencyList
#####

# This is for inference only - perform a single allocation for the whole adjacency list,
# with elements packed as closely together as possible.
#
# This is basically a CSR style format where the adacency list for vertex `i` is stored
# in the range `offsets[i]:(offsets[i+1] - 1)`.
# Adjacency lists are materialized lazily as views.
# NOTE: Offsets must be 1 longer than storage with `offsets[end] == length(storage) + 1`
"""
Space efficient implementation of an adjacency list stored in a CSC style representation.
All the individual adjacency lists are stored as a single dense vector using a vector of
offsets to retrieve individual neighbor lists.

This representation does not support mutation (i.e., index building) but is solely used
for querying.
"""
struct DenseAdjacencyList{T} <: AbstractAdjacencyList{T}
    storage::Vector{T}
    offsets::Vector{Int64}
end

# Implement the (read only) AdjacencyList API
_cannot_mutate() = error("Cannot modify a DenseAdjacencyList")

Base.push!(x::DenseAdjacencyList, v) = _cannot_mutate()
# The adjacency list must be constructed correctly so that the below rang is always inbounds.
function Base.getindex(x::DenseAdjacencyList, i)
    return Base.unsafe_view(x.storage, x.offsets[i]:(x.offsets[i + 1] - 1))
end

caninsert(x::DenseAdjacencyList, i) = false

Base.length(x::DenseAdjacencyList) = length(x.offsets) - 1
Base.length(x::DenseAdjacencyList, i) = x.offsets[i + 1] - x.offsets[i]
Base.empty!(x::DenseAdjacencyList) = _cannot_mutate()

function Base.iterate(x::DenseAdjacencyList, s = 1)
    return s > length(x) ? nothing : (@inbounds(x[s]), s + 1)
end

Base.copyto!(x::DenseAdjacencyList, args...) = _cannot_mutate()
