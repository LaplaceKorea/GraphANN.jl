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

raw(x::GraphANN.Euclidean) = x.vals

@testset "Testing Euclidean" begin
    x = zero(GraphANN.Euclidean{128,Float32})
    # Basic Definitions
    @test sizeof(x) == 128 * sizeof(Float32)
    @test sizeof(typeof(x)) == sizeof(x)
    @test length(x) == 128
    @test length(typeof(x)) == 128
    @test eltype(x) == Float32
    @test eltype(typeof(x)) == Float32

    # Indexing
    for i in 1:128
        @test iszero(x[i])
    end

    # Iteration
    @test all(iszero, x)

    # Basic Arithmetic
    x = GraphANN.Euclidean(ntuple(_ -> Float32(1), 64))
    y = GraphANN.Euclidean(ntuple(_ -> Float32(10), 64))
    @test all(isequal(11), x + y)
    @test all(isequal(9), y - x)

    # scalar division
    @test all(isequal(1), y / 10)

    # mapping
    @test all(isequal(100), map(i -> 10 * i, y))

    #####
    ##### SIMD schenangans
    #####

    # Instantiate the sentinel type to test some of the distance promotion logic.
    sentinel = GraphANN._Points.Sentinel{Nothing,Nothing}()
    distance_parameters = GraphANN._Points.distance_parameters
    distance_select = GraphANN._Points.distance_select

    @test distance_parameters(Nothing, Nothing) == sentinel
    @test_throws ArgumentError distance_select(sentinel, sentinel)

    # Now, get some actual distance parameters - make sure the `distance_select` does
    # its job and returns non-sentinel types
    parameters = distance_parameters(Float32, Float32)
    @test distance_select(parameters, sentinel) == parameters
    @test distance_select(sentinel, parameters) == parameters
    @test distance_select(parameters, parameters) == parameters

    # Do the same for assymetric types
    A = UInt8
    B = Float32
    @test !isa(
        distance_select(distance_parameters(A, B), distance_parameters(B, A)),
        GraphANN._Points.Sentinel,
    )
    @test !isa(
        distance_select(distance_parameters(B, A), distance_parameters(A, B)),
        GraphANN._Points.Sentinel,
    )

    # Test Casting - underlying data should be preserved
    a, b, c, d = ntuple(_ -> ntuple(_ -> rand(UInt8), 32), 4)
    x = GraphANN.Euclidean((a..., b..., c..., d...,))

    t = GraphANN._Points.cast(GraphANN.Euclidean{32,UInt8}, x)
    @test isa(t, NTuple{4, GraphANN.Euclidean{32,UInt8}})
    @test Tuple(t[1]) == a
    @test Tuple(t[2]) == b
    @test Tuple(t[3]) == c
    @test Tuple(t[4]) == d

    t = GraphANN._Points.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, NTuple{4, SIMD.Vec{32,UInt8}})
    @test Tuple(t[1]) == a
    @test Tuple(t[2]) == b
    @test Tuple(t[3]) == c
    @test Tuple(t[4]) == d

    # Test zero padding
    a = ntuple(_ -> rand(UInt8), 30)
    x = GraphANN.Euclidean(a)
    @test isa(x, GraphANN.Euclidean{30, UInt8})
    t = GraphANN._Points.cast(GraphANN.Euclidean{32,UInt8}, x)
    @test isa(t, Tuple{GraphANN.Euclidean{32, UInt8}})
    @test Tuple(t[1])[1:30] == a
    @test iszero(t[1][31])
    @test iszero(t[1][32])

    t = GraphANN._Points.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, Tuple{SIMD.Vec{32, UInt8}})
    @test Tuple(t[1])[1:30] == a
    @test iszero(t[1][31])
    @test iszero(t[1][32])

    # Now - time to test that SIMDWrap is doing its job.
    # The strategy is this:
    #
    # 1. Construct some tuples of UInt8
    # 2. Splat the tuples together to create a Euclidean
    # 3. SIMD wrap promoting to Float32
    # 4. Compare equality.
    a, b, c, d = ntuple(_ -> ntuple(_ -> rand(UInt8), 32), 4)

    x = GraphANN.Euclidean((a..., b..., c..., d...,))
    @test isa(x, GraphANN.Euclidean{128,UInt8})

    sx = GraphANN._Points.simd_wrap(SIMD.Vec{32,Float32}, x)
    @test isa(sx, GraphANN._Points.SIMDWrap)
    @test length(sx) == div(128, 32)
    @test length(sx[1]) == 32
    @test eltype(sx[1]) == Float32

    # Elements should remain equal after wrapping.
    @test Tuple(sx[1]) == a
    @test Tuple(sx[2]) == b
    @test Tuple(sx[3]) == c
    @test Tuple(sx[4]) == d
end

@testset "Testing Euclidean Calculations" begin
    # Test some basic constructors and stuff
    x = GraphANN.Euclidean{128,Float32}()
    @test length(x) == 128
    @test eltype(x) == Float32

    x = GraphANN.Euclidean{96,UInt8}()
    @test length(x) == 96
    @test eltype(x) == UInt8

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

    # Mixed types
    for i in 1:50000
        a = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
        b = GraphANN.Euclidean(ntuple(_ -> scale * rand(Float32), 128))

        @test isapprox(GraphANN.distance(a, b), euclidean_reference(raw(a), raw(b)))
    end

    # Test some unary operations.
    a = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
    b = a / 10
    @test raw(b) == raw(a) ./ 10

    c = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
    @test raw(a + c) == raw(a) .+ raw(c)
end

