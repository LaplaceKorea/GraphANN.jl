@testset "Testing SemiAtomicQueue" begin
    navailable = GraphANN._Prefetcher.Queue.navailable
    consume! = GraphANN._Prefetcher.Queue.consume!
    commit! = GraphANN._Prefetcher.Queue.commit!

    # Small buffer
    queue = GraphANN._Prefetcher.SemiAtomicQueue{Int}(5)
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

@testset "Testing multi-threading Atomic Queue" begin
    navailable = GraphANN._Prefetcher.Queue.navailable
    consume! = GraphANN._Prefetcher.Queue.consume!
    commit! = GraphANN._Prefetcher.Queue.commit!

    producer_pool = GraphANN.ThreadPool(1:1)
    consumer_pool = GraphANN.ThreadPool(2:2)

    # Make the queue smaller than the number of tokens to ensure that a wrap
    # around happens
    num_tokens = 10
    queue_size = 5
    queue = GraphANN._Prefetcher.SemiAtomicQueue{Int}(queue_size)

    # Create a producer-consumer relationship here - begin with the consumer first.
    dest = zeros(Int, num_tokens)
    consumer_handle = GraphANN.on_threads(consumer_pool, false) do
        index = 1
        while true
            sleep(0.05 * rand())
            collected = consume!(dest, index, queue)
            index += collected
            index > num_tokens && break
        end
        return nothing
    end

    producer_handle = GraphANN.on_threads(producer_pool, true) do
        for i in 1:num_tokens
            sleep(0.1 * rand())
            push!(queue, i)
            sleep(0.05 * rand())
            commit!(queue)
        end
        return nothing
    end

    # Wait for the consumer to finish
    wait(consumer_handle)
    @test dest == 1:num_tokens
end
