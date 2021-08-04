_increment(x::CartesianIndex{2}) = CartesianIndex((x[1] + 1, x[2]))
function convert_from_il_format(x::AbstractMatrix{T}; allocator = stdallocator) where {T}
    # Drop 1 from the length of the first dimension since the IL format uses 64-bit
    # integers for the lengths.
    adj = allocator(T, size(x,1) - 1, size(x,2))
    convert_from_il_format!(adj, x)
    return UniDirectedGraph{T}(SuperFlatAdjacencyList{T}(adj))
end

function convert_from_il_format!(adj::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T}
    dynamic_thread(CartesianIndices(adj), 8192) do i
        if i[1] == 1
            adj[i] = x[i]
        else
            adj[i] = x[_increment(i)] + one(T)
        end
    end
    return nothing
end

