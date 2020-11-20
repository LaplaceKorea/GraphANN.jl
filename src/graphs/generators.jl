#####
##### Generators
#####

function random_regular(
    ::Type{DefaultAdjacencyList{T}},
    nv,
    ne;
    max_edges = ne,
    slack = 1.00,
    allocator = stdallocator,
) where {T <: Integer}

    adj = Vector{Vector{T}}(undef, nv)
    _populate!(adj, ne, ceil(Int, slack * max_edges))
    @assert all(i -> isassigned(adj, i), 1:nv)

    return UniDirectedGraph{T}(DefaultAdjacencyList{T}(adj))
end

function random_regular(
    ::Type{FlatAdjacencyList{T}},
    nv,
    ne;
    max_edges = ne,
    slack = 1.00,
    allocator = stdallocator,
) where {T <: Integer}

    # Allocate destination space
    adj = allocator(T, ceil(Int, slack * max_edges), nv)
    adj .= zero(T)

    _populate!(adj, ne)
    lengths = fill(T(ne), nv)
    return UniDirectedGraph{T}(FlatAdjacencyList{T}(adj, lengths))
end

# I don't really know what to call these ... helper functions??
function _populate!(A::Matrix{T}, ne) where {T}
    tls = ThreadLocal(T[])
    nv = size(A, 2)

    dynamic_thread(1:nv, 128) do i
        storage = tls[]
        empty!(storage)

        while length(storage) < ne
            i = rand(1:nv)
            if (i != col) && !in(i, storage)
                push!(storage, i)
            end
        end
        sort!(storage)
        @views A[1:ne, col] .= storage
    end
end

function _populate!(A::AbstractVector{<:AbstractVector{T}}, ne, max_edges) where {T}
    nv = length(A)

    Threads.@threads for col in 1:nv
        list = T[]
        sizehint!(list, max_edges)
        while length(list) < ne
            i = rand(1:nv)
            if (i != col) && !in(i, list)
                push!(list, i)
            end
        end
        sort!(list)
        A[col] = list
    end
    return A
end

