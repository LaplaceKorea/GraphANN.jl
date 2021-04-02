module _Graphs

# local deps
using .._Base

# deps
using LightGraphs: LightGraphs
using ProgressMeter: ProgressMeter
import UnPack: @unpack

# explicit imports
import LightGraphs.SimpleGraphs.fadj

# Top level file - include the implementation files.
export DefaultAdjacencyList, FlatAdjacencyList, DenseAdjacencyList
include("adjacency.jl")

export UniDirectedGraph
include("unidirected.jl")

export random_regular
include("generators.jl")

function vertices_in_radius(
    g::LightGraphs.AbstractGraph, s::Integer, hops = 5; max_vertices = nothing
)
    T = eltype(g)
    visited = RobinSet(T(s))
    # Double buffer search frontiers
    # This will save a bit on allocations
    frontier = [T(s)]
    next_frontier = T[]
    for i in 1:hops
        println("Working on frontier: $i")
        empty!(next_frontier)
        breakout = false

        for u in frontier
            for v in LightGraphs.outneighbors(g, u)
                in(v, visited) && continue
                push!(next_frontier, v)
                push!(visited, v)
            end

            # Maybe abort.
            if max_vertices !== nothing && length(visited) > max_vertices
                breakout = true
                break
            end
        end
        breakout && break

        # Swap frontier buffers
        frontier, next_frontier = next_frontier, frontier
    end
    return visited
end

end #module
