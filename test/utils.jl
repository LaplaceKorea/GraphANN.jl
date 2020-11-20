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
    @test in(10, x) == false

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

@testset "Testing BatchedRange" begin
    range = 1:100
    x = GraphANN.BatchedRange(range, 10)
    @test length(x) == 10
    @test x[1] == 1:10
    @test x[2] == 11:20
    @test x[10] == 91:100

    @test_throws BoundsError x[0]
    @test_throws BoundsError x[11]

    # Make sure the the last batch is handled correctly.
    range = 1:10
    x = GraphANN.BatchedRange(range, 3)
    @test x[1] == 1:3
    @test x[2] == 4:6
    @test x[3] == 7:9
    @test x[4] == 10:10
    # iteration
    @test collect(x) == [1:3, 4:6, 7:9, 10:10]

    # Finally, make sure affine translations still work.
    range = 10:2:30
    x = GraphANN.BatchedRange(range, 4)
    @test x[1] == 10:2:16
    @test x[2] == 18:2:24
    @test x[3] == 26:2:30

    # Test iteration
    @test collect(x) == [10:2:16, 18:2:24, 26:2:30]
end
