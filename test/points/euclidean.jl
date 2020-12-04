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
end
