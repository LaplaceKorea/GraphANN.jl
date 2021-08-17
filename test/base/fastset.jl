@testset "FasetSet" begin
    x = GraphANN.FastSet{Int}()
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

    # i = pop!(x)
    empty!(x)
    @test length(x) == 0

    push!(x, 1)
    push!(x, 2)
    @test length(x) == 2
    empty!(x)
    @test length(x) == 0
    @test in(1, x) == false
    @test in(2, x) == false
end
