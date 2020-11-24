@testset "Testing SemiAtomicQueue" begin
    navailable = GraphANN.Queue.navailable
    consume! = GraphANN.Queue.consume!
    commit! = GraphANN.Queue.commit!

    # Small buffer
    queue = GraphANN.SemiAtomicQueue{Int}(5)
    @test navailable(queue) == 0
    push!(queue, 1)
    push!(queue, 2)

    # Updates aren't available until commit! is called.
    @test navailable(queue) == 0

    # Make sure `consume!` doesn't do anything.
    dest = [0,0]
    count = consume!(dest, 1, queue)
    @test dest == [0,0]
    @test count == 0

    commit!(queue)
    @test navailable(queue) == 2
    count = consume!(dest, 1, queue)
    @test dest == [1,2]
    @test count == 2
    @test navailable(queue) == 0

    # Try adding 4 items - all should get queued.
    # `push!` will return `true` if the item was successfully added
    @test push!(queue, 1) == true
    @test push!(queue, 2) == true
    @test push!(queue, 3) == true
    @test push!(queue, 4) == true

    @test navailable(queue) == 0
    commit!(queue)
    @test navailable(queue) == 4

    @test push!(queue, 5) == false
    @test navailable(queue) == 4
    commit!(queue)
    @test navailable(queue) == 4

    # Consume into a different starting location
    dest = [0,0,0,0,0]
    count = consume!(dest, 2, queue)
    @test count == 4
    @test dest == [0, 1, 2, 3, 4]
    @test navailable(queue) == 0
    @test push!(queue, 5) == true
end
