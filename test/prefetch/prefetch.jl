@testset "Testing Prefetch Staging" begin
    # For this test, since we only assume having access to 2 threads during the unit
    # test set, we make both the producer and consumer pool run on threads 1 and 2.
    #
    # The routines that run on each thread will need a `yield` statement to ensure that
    # the schedule can switch between the two.
    producer_pool = GraphANN.ThreadPool(1:2)
    consumer_pool = GraphANN.ThreadPool(1:2)

    # One queue for each producer
    queue1 = GraphANN._Prefetcher.SemiAtomicQueue{Int}(100)
    queue2 = GraphANN._Prefetcher.SemiAtomicQueue{Int}(100)

    queues = Dict(i => (queue1, queue2)[i] for i in producer_pool)
    staging = GraphANN._Prefetcher.Staging(queues)

    tokens_per_thread = 1000
    # One destination per consumer
    destinations = [Int[], Int[]]
    stop_signals = [Ref(false), Ref(false)]
    tokens = [1:(1 + tokens_per_thread), 100000:(100000 + tokens_per_thread)]

    consumer_handles = GraphANN.on_threads(consumer_pool, false) do
        tid = Threads.threadid()
        destination = destinations[tid]
        stop_signal = stop_signals[tid]

        buffer = Int[]
        num_to_acquire = 5

        while !stop_signal[]
            GraphANN._Prefetcher.acquire!(staging, buffer, num_to_acquire)
            append!(destination, buffer)

            sleep(0.0025 * rand())
            yield()
        end
    end

    # Producers
    producer_handles = GraphANN.on_threads(producer_pool, true) do
        tid = Threads.threadid()
        items = tokens[tid]
        queue = queues[tid]

        for item in items
            push!(queue, item)

            # Only update once in a while
            if rand() <= 0.25
                GraphANN._Prefetcher.commit!(queue)
            end

            sleep(0.002 * rand())
            yield()
        end
        GraphANN._Prefetcher.commit!(queue)
    end

    # Wait for the queues to be drained
    sleep(4)
    foreach(i -> i[] = true, stop_signals)
    wait(consumer_handles)

    # Now, we make sure that all tokens made it through successfully without repeats
    # or losses
    expected = sort(reduce(vcat, tokens))
    result = sort(reduce(vcat, destinations))
    @test result == expected
end
