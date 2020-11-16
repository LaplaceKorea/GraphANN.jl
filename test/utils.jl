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

@testset "Testing RobinSet" begin
    x = GraphANN.RobinSet{Int}()
    @test length(x) == 0
    push!(x, 10)
    @test length(x) == 1
    push!(x, 20)
    @test length(x) == 2

    @test in(10, x) == true
    @test in(30, x) == false
    @test in(20, x) == true
    @test in(0, x) == false

    # iterator
    @test sort(collect(x)) == [10, 20]

    # deletion
    delete!(x, 10)
    @test length(x) == 1
    @test in(20, x) == true
    @test in(10, x) == true

    i = pop!(x)
    @test length(x) == 0

    push!(x, 1)
    push!(x, 2)
    @test length(x) == 2
    empty!(x)
    @test length(x) == 0
    @test in(1, x) == false
    @test in(2, x) == false
end
