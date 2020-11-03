"""
    MetaGraph{G,D}

Grouping of a graph of type `G` and corresponding vertex data points of type `D`.
"""
struct MetaGraph{G,D}
    graph::G
    data::D
end

# Construct a dummy test graph for testing purposes.
"""
    random_graph(num_vertices, [edges])

Construct a randomly connect graph with `num_vertices` number of vertices.
Each vertex will have `edges` neighbors.
"""
function random_graph(num_vertices, num_edges =  2 * ceil(Int, log(num_vertices)))
    # Sanity check of inputs
    if num_edges > num_vertices
        err = ArgumentError("""
            Number of neighbors ($num_edges) is greater than number of vertices ($num_vertices)!
            """
        )
        throw(err)
    end

    graph = SimpleDiGraph(num_vertices)
    for v in vertices(graph)
        # Get a reference to the adjacency list
        neighbors = outneighbors(graph, v)
        while length(neighbors) < num_edges
            u = rand(vertices(graph))

            # if we don't have this neighbor yet, add it to the graph.
            (v == u || in(u, neighbors)) && continue
            add_edge!(graph, v, u)
        end
    end
    return graph
end
