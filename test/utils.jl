@testset "Testing Neighbor" begin
    Neighbor = GraphANN.Neighbor

    a = Neighbor(1, 1.0)
    b = Neighbor(1, 2.0)
    @test GraphANN.getid(a) == 1
    @test GraphANN.getid(b) == 1
    @test GraphANN.getdistance(a) == 1.0
    @test GraphANN.getdistance(b) == 2.0

    @test GraphANN.idequal(a, b)
    c = Neighbor(2, 5.0)
    @test GraphANN.idequal(a, c) == false

    # Ordering
    @test Neighbor(1, 1.0) < Neighbor(2, 2.0)
    @test Neighbor(10, 5.0) > Neighbor(40, 1.2)

    # Total Ordering
    @test Neighbor(1, 1.0) < Neighbor(2, 1.0)
    @test Neighbor(2, 1.0) > Neighbor(1, 1.0)
end
