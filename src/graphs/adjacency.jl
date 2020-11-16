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

# Define the interface
abstract type AbstractAdjacencyList{T <: Integer} end
caninsert(x::T) where {T <: AbstractAdjacencyList} = caninsert(T)

#####
##### Default implementation
#####

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

Base.iterate(x::DefaultAdjacencyList) = iterate(x.fadj)
Base.iterate(x::DefaultAdjacencyList, s) = iterate(x.fadj, s)

function Base.copyto!(x::DefaultAdjacencyList, v, A::AbstractArray)
    list = x[v]
    resize!(list, length(A))
    copyto!(list, A)
    sort!(list; alg = Base.QuickSort)
end

#####
##### Flat Implementation
#####

struct FlatAdjacencyList{T} <: AbstractAdjacencyList{T}
    adj::Matrix{T}
    # Store how many neighbors each vertex actually has.
    lengths::Vector{T}

    # -- Inner constructor to avoid ambiguity
    function FlatAdjacencyList{T}(adj::Matrix{T}, lengths::Vector{T}) where {T}
        return new{T}(adj, lengths)
    end
end

function FlatAdjacencyList{T}(nv::Integer, max_degree::Integer) where {T}
    adj = zeros(T, max_degree, nv)
    lengths = zeros(T, nv)
    return FlatAdjacencyList{T}(adj, lengths)
end

_max_degree(x::FlatAdjacencyList) = size(x.adj, 1)
Base.length(x::FlatAdjacencyList) = size(x.adj, 2)
Base.length(x::FlatAdjacencyList, i) = x.lengths[i]
Base.empty!(x::FlatAdjacencyList, i) = (x.lengths[i] = 0)

Base.push!(x::FlatAdjacencyList, v) = error("Cannot yet push to a FlatAdjacencyList!")

# Use `unsafe_view` because `lenght` will bounds check for us.
Base.getindex(x::FlatAdjacencyList, i) = Base.unsafe_view(x.adj, 1:length(x, i), i)

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

function Base.copyto!(x::FlatAdjacencyList, v, A::AbstractArray)
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

