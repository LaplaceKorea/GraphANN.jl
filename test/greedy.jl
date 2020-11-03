@testset "Testing GreedySearch Methods" begin
    # Test some properties of `Neighbor`
    @test Neighbor(1, 1.0) < Neighbor(1, 2.0)
    @test Neighbor(10, 1.0) < Neighbor(1, 2.0)
    @test Neighbor(1, 1.0) < Neighbor(10, 2.0)

    x = GraphANN.GreedySearch(2)
    GraphANN.pushcandidate!(x, Neighbor(1, 0.0))

    # Use `egal` for comparison in case we want to overload `==`
    @test maximum(x) === Neighbor(1, 0.0)
    GraphANN.pushcandidate!(x, Neighbor(2, 10.0))
    @test maximum(x) === Neighbor(2, 10.0)

    GraphANN.pushcandidate!(x, Neighbor(3, -10.0))
    @test maximum(x) === Neighbor(2, 10.0)

    GraphANN.reduce!(x)
    @test maximum(x) === Neighbor(1, 0.0)
end
