@testset "Testing Pruner" begin
    # Make this definition for convenience
    Pruner = GraphANN.Pruner

    x = [3,2,1]
    pruner = Pruner{eltype(x)}()

    @test length(pruner) == 0

    # Add all the items in `x` to the pruner.
    GraphANN.initialize!(pruner, x)
    @test length(pruner) == 3
    @test GraphANN.ispruned(pruner, 1) == false
    @test GraphANN.ispruned(pruner, 2) == false
    @test GraphANN.ispruned(pruner, 3) == false
    @test collect(pruner) == x

    # Does sorting work?
    sort!(pruner)
    @test collect(pruner) == sort(x)

    # Prune out the middle element
    GraphANN.prune!(pruner, 2)
    @test GraphANN.ispruned(pruner, 1) == false
    @test GraphANN.ispruned(pruner, 2) == true
    @test GraphANN.ispruned(pruner, 3) == false
    @test collect(pruner) == [1,3]

    # Prune out other items
    GraphANN.prune!(pruner, 3)
    @test GraphANN.ispruned(pruner, 1) == false
    @test GraphANN.ispruned(pruner, 2) == true
    @test GraphANN.ispruned(pruner, 3) == true
    @test collect(pruner) == [1]

    GraphANN.prune!(pruner, 1)
    @test GraphANN.ispruned(pruner, 1) == true
    @test GraphANN.ispruned(pruner, 2) == true
    @test GraphANN.ispruned(pruner, 3) == true
    @test collect(pruner) == []

    # Try again, but now use the filter function.
    pruner = GraphANN.Pruner{Int}()
    GraphANN.initialize!(pruner, 1:100)
    GraphANN.prune!(isodd, pruner)
    @test collect(pruner) == 2:2:100
end
