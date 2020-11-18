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

    # Finally, make sure affine translations still work.
    range = 10:2:30
    x = GraphANN.BatchedRange(range, 4)
    @test x[1] == 10:2:16
    @test x[2] == 18:2:24
    @test x[3] == 26:2:30
end
