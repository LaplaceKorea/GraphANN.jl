@testset "Testing partition!" begin
    x = collect(1:10000)
    shuffle!(x)
    y = deepcopy(x)

    util = GraphANN.PartitionUtil{eltype(x)}()

    pred = i -> i > 5000
    index = GraphANN.partition!(pred, x, util; executor = GraphANN.single_thread)
    @test all(pred, view(x, 1:(index - 1)))
    @test all(!pred, view(x, index:lastindex(x)))

    # Make sure that multi-threading yields the exact same result.
    index = GraphANN.partition!(i -> i > 5000, y, util; executor = GraphANN.dynamic_thread)
    @test y == x

    ### Now, make sure we get the correct results at the edge cases.
    # No elements satisfy pred.
    pred = i -> i > (2 * length(x))
    index = GraphANN.partition!(pred, x, util; executor = GraphANN.single_thread)
    @test index == 1
    @test all(pred, view(x, 1:(index - 1)))
    @test all(!pred, view(x, index:lastindex(x)))

    index = GraphANN.partition!(pred, y, util; executor = GraphANN.dynamic_thread)
    @test y == x

    # All elements satisfy pred.
    pred = i -> i < 1_000_000
    index = GraphANN.partition!(pred, x, util; executor = GraphANN.single_thread)
    @test index == length(x) + 1
    @test all(pred, view(x, 1:(index - 1)))
    @test all(!pred, view(x, index:lastindex(x)))

    index = GraphANN.partition!(pred, y, util; executor = GraphANN.dynamic_thread)
    @test y == x
end
