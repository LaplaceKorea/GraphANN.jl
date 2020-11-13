@testset "Testing UniDirectedGraph" begin
    # For testing, I've added a `using LightGraphs` to `runtests.jl` to save on typing ...
    g = GraphANN.UniDirectedGraph{UInt32}()
    @test nv(g) == 0
    @test ne(g) == 0
    n = add_vertex!(g)

    @test n == 1
    @test nv(g) == 1
    @test ne(g) == 0

    n = add_vertex!(g)
    @test n == 2
    @test nv(g) == 2
    @test ne(g) == 0
    @test vertices(g) == 1:2

    @test has_vertex(g, 1)
    @test has_vertex(g, 2)
    @test has_vertex(g, 0) == false
    @test has_vertex(g, 3) == false

    degree = add_edge!(g, 1, 2)
    @test degree == 1
    @test ne(g) == 1
    @test has_edge(g, 1, 2)
    @test has_edge(g, 2, 1) == false

    @test outneighbors(g, 1) == [2]
    @test outneighbors(g, 2) == []
    @test collect(inneighbors(g, 1)) == []
    @test collect(inneighbors(g, 2)) == [1]

    # Test inference properties of edge iter
    @test eltype(edges(g)) <: LightGraphs.SimpleGraphs.SimpleDiGraphEdge

    # Moar Graphs
    g = GraphANN.UniDirectedGraph(10)
    add_edge!(g, 1, 9)
    add_edge!(g, 1, 5)

    # Make sure adjacency lists are sorted
    @test outneighbors(g, 1) == [5, 9]
    add_edge!(g, 1, 7)
    @test outneighbors(g, 1) == [5, 7, 9]
    add_edge!(g, 1, 2)
    @test outneighbors(g, 1) == [2, 5, 7, 9]
    add_edge!(g, 1, 10)
    @test outneighbors(g, 1) == [2, 5, 7, 9, 10]

    # Make sure our copy function thing works
    nn = [1,2,3,4,5]
    GraphANN.sorted_copy!(g, 1, nn)
    @test outneighbors(g, 1) == nn

    @test collect(inneighbors(g, 5)) == [1]
    @test has_edge(g, 10, 5) == false
end
