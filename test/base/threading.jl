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

# Customize `threadcopy` to NOT deep copy.
struct ArrayWrapper{N,T} <: AbstractArray{N,T}
    val::Array{N,T}
end
GraphANN._Base.threadcopy(x::ArrayWrapper) = x

@testset "Testing Threading" begin
    @testset "ThreadPool Methods" begin
        @test GraphANN.allthreads() === GraphANN.ThreadPool(Base.OneTo(Threads.nthreads()))
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

    @testset "Dynamic Threading" begin
        lock = ReentrantLock()
        stack = Int[]
        threads = Int[]
        pool = GraphANN.ThreadPool(Base.OneTo(2))

        # Thread across a collection - make sure that work is actually being distributed.
        range = 1:100
        GraphANN.dynamic_thread(pool, range) do i
            sleep(0.01 * rand())
            Base.@lock lock begin
                push!(stack, i)
                push!(threads, Threads.threadid())
            end
        end

        # The stack should contain all the integers in the provided range, but they
        # shouldn't be in order.
        @test sort(stack) == range
        @test !issorted(stack)
        @test count(isequal(1), threads) >= 0.3 * length(range)
        @test count(isequal(2), threads) >= 0.3 * length(range)

        # Automatically apply thread pool.
        empty!(stack)
        empty!(threads)
        GraphANN.dynamic_thread(pool, range) do i
            sleep(0.01 * rand())
            Base.@lock lock begin
                push!(stack, i)
                push!(threads, Threads.threadid())
            end
        end
        @test sort(stack) == range
        @test !issorted(stack)
        # At least two distinct threads present.
        @test length(unique(stack)) >= 2

        # Try batching
        empty!(stack)
        empty!(threads)
        GraphANN.dynamic_thread(pool, range, 10) do i
            sleep(0.01 * rand())
            Base.@lock lock begin
                push!(stack, i)
                push!(threads, Threads.threadid())
            end
        end
        @test sort(stack) == range
        @test !issorted(stack)
        @test count(isequal(1), threads) >= 0.3 * length(range)
        @test count(isequal(2), threads) >= 0.3 * length(range)

        # Now, make sure that `single_thread` has a similar signature.
        empty!(stack)
        empty!(threads)
        GraphANN.single_thread(pool, range, 10) do i
            Base.@lock lock begin
                push!(stack, i)
                push!(threads, Threads.threadid())
            end
        end
        # Result should be sorted since there's no parallelism.
        @test issorted(stack)
        @test stack == range
        # only one thread present.
        @test length(unique(threads)) == 1
    end

    @testset "Threadcopy" begin
        threadcopy = GraphANN._Base.threadcopy
        # Normal behavior is to deep copy.
        x = Int[1,2,3]
        y = threadcopy(x)
        @test x == y
        @test x !== y

        # See if extending `threadcopy` works.
        A = ArrayWrapper(x)
        B = threadcopy(A)
        @test A === B
        @test A.val === B.val

        # Make sure NamedTuples behave correctly.
        x = Int[1,2,3]
        y = ArrayWrapper(Int[1,2,3])
        original = (x = x, y = y)
        copy = threadcopy(original)

        @test original.x == copy.x
        @test original.x !== copy.x
        @test original.y === copy.y
    end

    @testset "ThreadLocal" begin
        pool = GraphANN.ThreadPool(1:2)
        # Keyword Only Constructor
        tls = GraphANN.ThreadLocal(pool; a = 10, b = "hello world")
        @test isa(GraphANN._Base.getall(tls), Vector{<:NamedTuple})
        @test length(tls.values) == length(pool)
        @test tls[].a == 10
        @test tls[].b == "hello world"
        @test tls[1].a == 10
        @test tls[1].b == "hello world"
        @test tls[2].a == 10
        @test tls[2].b == "hello world"

        # Also just try the normal copying constructor
        tls = GraphANN.ThreadLocal(pool, Int[1,2,3])
        values = GraphANN.getall(tls)
        @test length(values) == 2
        # test for deep copy
        @test values[1] == values[2]
        @test values[1] !== values[2]

        # Make sure `threadcopy` is called and test indexing for OneTo works
        x = Int[1,2,3]
        y = ArrayWrapper(Int[1,2,3])
        tls = GraphANN.ThreadLocal(; x = x, y = y)
        @test length(GraphANN.getall(tls)) >= 2
        @test tls[1].x == tls[2].x
        @test tls[1].x !== tls[2].x
        @test tls[1].y === tls[2].y

        @test GraphANN.getlocal(tls).x === tls[1].x

        # Misc other methods
        nt = (a = 1, b = 2)
        v = only(GraphANN._Base.getall(nt))
        @test v == nt
        x = Int[1,2,3]
        @test GraphANN.getlocal(x) === x
    end
end
