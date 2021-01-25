@testset "Testing Spans" begin
    A = collect(Float32.(1:10))

    x = GraphANN.Span(pointer(A), length(A))
    @test length(x) == length(A)
    for i in 1:length(x)
        @test x[i] == i
    end

    @test eltype(x) == Float32
    @test Base.IndexStyle(x) == Base.IndexLinear()
    @test size(x) == size(A)
    @test pointer(x) == pointer(A)

    @test_throws BoundsError x[0]
    @test_throws BoundsError x[length(x) + 1]

    # Can we store tuples of `isbits` types as well?
    A = [(i, i+1, i+2, i+3) for i in 1:10]
    x = GraphANN.Span(pointer(A, 2), length(A) - 2)
    @test x[1] == (2, 3, 4, 5)
    @test x[end] == (9, 10, 11, 12)
    x[1] = (0, 0, 0, 0)
    @test A[2] == (0, 0, 0, 0)

    # Test that code vectorizes properly.
    # If it DOES vectorize, then we should see some unrolled loops in the assembly
    # code for the broadcasted add function.
    io = IOBuffer(; read = true, write = true)
    code_native(
        io,
        GraphANN._Base.broadcast_add,
        Tuple{GraphANN.Span{Float32}, GraphANN.Span{Float32}};
        syntax = :intel,
    )

    seekstart(io)
    vcount = 0
    unroll_count = 4
    for ln in eachline(io)
        ln = strip(ln)
        if startswith(ln, "vadd")
            vcount += 1
        else
            vcount = 0
        end
        # Look for unrolled vectors
        vcount == unroll_count && break
    end

    # Julia doesn't agressively compile in test mode.
    # That's annoying.
    @test_broken vcount == unroll_count

end
