# Are we running with enough threads?
if Threads.nthreads() == 1
    error("Please run tests with at least two threads!")
end

# Use this as a background task to make sure that we can actually keep tasks running
# in the background.
function spin(
    A::Vector{T},
    channel::Channel{T},
    stop::Ref{Bool},
    resp::Ref{Bool}
) where {T}
    empty!(A)
    while !stop[]
        isready(channel) && push!(A, take!(channel))
        sleep(0.01)
    end
    resp[] = true
    return nothing
end

@testset "Testing Threading" begin
    @testset "ThreadPool Methods" begin
        @test GraphANN.allthreads() == GraphANN.ThreadPool(1:Threads.nthreads())
        pool = GraphANN.ThreadPool(2:4)
        @test length(pool) == 3
        @test first(pool) == 2
        @test firstindex(pool) == 1

        @test eachindex(pool) == 1:3
        # iterations
        @test collect(pool) == [2,3,4]

        pool = GraphANN.ThreadPool([10, 20, 30, 40])
        @test length(pool) == 4
        @test first(pool) == 10
        @test collect(pool) == [10, 20, 30, 40]
    end

    @testset "Background Threads" begin
        p1 = GraphANN.ThreadPool(1:1)
        p2 = GraphANN.ThreadPool(2:2)

        A = Int[]
        channel = Channel{Int}(100)
        stop = Ref(false)
        resp = Ref(false)

        # First, make sure that exception handling works
        # We purposely call `spin` with too few arguments which will result in an error.
        # Since `on_threads` returns a handle to the spawned tasks, we can inspect those
        # tasks to see if an error occured.
        runner = GraphANN.on_threads(p1, false) do
            spin(A, channel, stop)
        end
        @test_throws TaskFailedException wait(runner)

        # Pass `false` as the second argument to avoid blocking.
        GraphANN.on_threads(p1, false) do
            spin(A, channel, stop, resp)
        end

        # At this point, nothing should have been added to A
        @test isempty(A)
        @test resp[] == false

        # Spawn another task on thread 2.
        GraphANN.on_threads(p2) do
            for i in 1:10
                put!(channel, i)
            end
        end
        sleep(1)

        @test length(A) == 10
        @test resp[] == false

        # Stop the background task
        stop[] = true
        sleep(0.1)
        @test length(A) == 10
        @test resp[] == true
        @test A == 1:10
    end

    @testset "Thread Assignment" begin
        # Are we actually assigning work to the correct number of threads?
        p1 = GraphANN.ThreadPool(1:1)
        p2 = GraphANN.ThreadPool(2:2)

        id = Ref{Int}(0)
        handle = GraphANN.on_threads(p1) do
            id[] = Threads.threadid()
        end
        @test length(handle) == 1
        @test id[] == 1

        handle = GraphANN.on_threads(p2) do
            id[] = Threads.threadid()
        end
        @test length(handle) == 1
        @test id[] == 2
    end
end
