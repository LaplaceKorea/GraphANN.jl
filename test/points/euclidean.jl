# Reference implementation
# This isn't exactly meant to be robust - just be pretty obcious
function euclidean_reference(A, B)
    s = zero(Float64)
    for (a,b) in zip(A,B)
        c = Float64(a) - Float64(b)
        s += c^2
    end
    return s
end

@testset "Testing Euclidean Calculations" begin
    # Test some basic constructors and stuff
    x = GraphANN.Euclidean{128,Float32}()
    @test length(x) == 128
    @test eltype(x) == Float32
    @test isa(GraphANN.raw(x), NTuple{128,Float32})

    x = GraphANN.Euclidean{96,UInt8}()
    @test length(x) == 96
    @test eltype(x) == UInt8
    @test isa(GraphANN.raw(x), NTuple{96,UInt8})

    raw = GraphANN.raw

    # Lets do some distance calculations
    scale = 100
    for i in 1:50000
        a = GraphANN.Euclidean(ntuple(_ -> scale * randn(Float32), 128))
        b = GraphANN.Euclidean(ntuple(_ -> scale * randn(Float32), 128))

        @test isapprox(GraphANN.distance(a, b), euclidean_reference(raw(a), raw(b)))
    end

    for i in 1:50000
        a = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
        b = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))

        @test isapprox(GraphANN.distance(a, b), euclidean_reference(raw(a), raw(b)))
    end

    # Test some unary operations.
    a = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
    b = a / 10
    @test raw(b) == GraphANN.raw(a) ./ 10

    c = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
    @test raw(a + c) == raw(a) .+ raw(c)

    ### Test copying

    # N.B.: Need to make `A` fairly large to ensure that the base pointer for `A`
    # is aligned to a cacheline boundary.
    # TODO: How do we ensure this in general?
    A = Vector{GraphANN.Euclidean{128,UInt8}}(undef, 100)
    z = eltype(A)()
    x = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
    A .= Ref(z)

    A[3] = x
    @test A[2] == z
    @test A[3] == x
    @test A[4] == z

    @test A[5] == z
    @test A[6] == z
    @test A[7] == z
    GraphANN.fast_copyto!(pointer(A, 6), pointer(A, 3))
    @test A[5] == z
    @test A[6] == x
    @test A[7] == z
end
