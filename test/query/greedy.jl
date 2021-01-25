@testset "Testing GreedySearch Methods" begin
    # Test some properties of `Neighbor`
    x = GraphANN.GreedySearch(2; costtype = Float64, idtype = Int64)
    GraphANN.Algorithms.pushcandidate!(x, Neighbor(x, 1, 0.0))

    # Test `Neighbor` constructor for the `GreedySearch` type.
    @test isa(GraphANN.Neighbor(x, 1, 1.0), GraphANN.Neighbor{Int64,Float64})
    # convert id types
    @test isa(GraphANN.Neighbor(x, UInt32(1.0), 1.0), GraphANN.Neighbor{Int64,Float64})
    # no implicit conversion of distance types
    @test_throws MethodError GraphANN.Neighbor(x, 1.0, Float32(10))

    # Use `egal` for comparison in case we want to overload `==`
    @test maximum(x) === Neighbor(x, 1, 0.0)
    GraphANN.Algorithms.pushcandidate!(x, Neighbor(x, 2, 10.0))
    @test maximum(x) === Neighbor(x, 2, 10.0)

    GraphANN.Algorithms.pushcandidate!(x, Neighbor(x, 3, -10.0))
    @test maximum(x) === Neighbor(x, 2, 10.0)

    GraphANN.Algorithms.reduce!(x)
    @test maximum(x) === Neighbor(x, 1, 0.0)
end
