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

@testset "Testing Euclidean" begin
    Euclidean = GraphANN.Euclidean
    x = zero(Euclidean{128,Float32})
    # Basic Definitions
    @test sizeof(x) == 128 * sizeof(Float32)
    @test sizeof(typeof(x)) == sizeof(x)
    @test length(x) == 128
    @test length(typeof(x)) == 128
    @test eltype(x) == Float32
    @test eltype(typeof(x)) == Float32
    @test x == x
    @test transpose(x) === x

    # Indexing
    for i in 1:128
        @test iszero(x[i])
    end

    # Iteration
    @test all(iszero, x)

    # Basic Arithmetic
    x = Euclidean{64}(Float32(1))
    y = Euclidean{64}(Float32(10))
    @test all(isequal(11), x + y)
    @test Euclidean{64}(11) == x + y
    @test all(isequal(9), y - x)
    @test Euclidean{64}(9) == y - x

    # scalar division
    @test all(isequal(1), y / 10)

    # mapping
    @test all(isequal(100), map(i -> 10 * i, y))

    # broadcasting.
    x = [Euclidean{16}(i) for i in 1:10]
    y = Euclidean{16}(100)
    z = x .+ y
    expected = [Euclidean{16}(i + 100) for i in 1:10]
    @test z == expected

    # conversion
    x = rand(Euclidean{64,UInt8})
    y = convert(Euclidean{64,Float32}, x)

    @test isa(y, Euclidean{64,Float32})
    # test equality two different ways
    @test x == y
    @test x !== y
    for i in 1:length(x)
        @test x[i] == y[i]
    end

    # identity conversion
    @test x == convert(Euclidean{64,UInt8}, x)
    @test x === convert(Euclidean{64,UInt8}, x)

    # conversion of short Euclideans to SIMD.Vec
    x = rand(Euclidean{8,Float32})
    for T in [Float32, Float64]
        y = convert(SIMD.Vec{8,T}, x)
        for i in 1:length(x)
            @test x[i] == y[i]
        end
    end

    #####
    ##### SIMD schenangans
    #####

    # Test Casting - underlying data should be preserved
    _merge = GraphANN._Points._merge
    a, b, c, d = ntuple(_ -> rand(Euclidean{32,UInt8}), 4)
    x = Euclidean(_merge(a, b, c, d))

    t = GraphANN._Points.cast(Euclidean{32,UInt8}, x)
    @test isa(t, NTuple{4, Euclidean{32,UInt8}})
    @test t == (a, b, c, d)

    # For comparison with SIMD.Vec, convert both to Tuples
    t = GraphANN._Points.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, NTuple{4, SIMD.Vec{32,UInt8}})
    @test Tuple(t[1]) == Tuple(a)
    @test Tuple(t[2]) == Tuple(b)
    @test Tuple(t[3]) == Tuple(c)
    @test Tuple(t[4]) == Tuple(d)

    # Test zero padding
    a = ntuple(_ -> rand(UInt8), 30)
    x = Euclidean(a)
    @test isa(x, Euclidean{30, UInt8})
    t = GraphANN._Points.cast(Euclidean{32,UInt8}, x)
    @test isa(t, Tuple{Euclidean{32, UInt8}})
    @test Tuple(t[1])[1:30] == a
    @test iszero(t[1][31])
    @test iszero(t[1][32])

    t = GraphANN._Points.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, Tuple{SIMD.Vec{32, UInt8}})
    @test Tuple(t[1])[1:30] == a
    @test iszero(t[1][31])
    @test iszero(t[1][32])

    # Now - time to test that EagerWrap is doing its job.
    # The strategy is this:
    #
    # 1. Construct some tuples of UInt8
    # 2. Splat the tuples together to create a Euclidean
    # 3. SIMD wrap promoting to Float32
    # 4. Compare equality.
    a, b, c, d = ntuple(_ -> rand(Euclidean{32,UInt8}), 4)
    x = Euclidean(_merge(a, b, c, d))
    @test isa(x, Euclidean{128,UInt8})

    sx = GraphANN._Points.EagerWrap{SIMD.Vec{32,Float32}}(x)
    @test isa(sx, GraphANN._Points.EagerWrap)
    @test length(sx) == div(128, 32)
    @test length(sx[1]) == 32
    @test eltype(sx[1]) == Float32

    # Elements should remain equal after wrapping.
    @test Tuple(sx[1]) == Tuple(a)
    @test Tuple(sx[2]) == Tuple(b)
    @test Tuple(sx[3]) == Tuple(c)
    @test Tuple(sx[4]) == Tuple(d)

    # Promotion Logic
    find_distance_type = GraphANN._Points.find_distance_type

    # First, test the fallback
    @test find_distance_type(Float32, Int) == promote_type(Float32, Int)
    @test find_distance_type(Int, Float32) == promote_type(Float32, Int)

    # now, test our specializations for UInt8 work
    @test find_distance_type(UInt8, UInt8) == Int16
    @test find_distance_type(UInt8, Int16) == Int16
    @test find_distance_type(Int16, UInt8) == Int16

    # Manually try out some promotions - especially check for inference.
    simd_type = GraphANN._Points.simd_type

    # Full AVX-512 vector
    F32 = Euclidean{128,Float32}
    F64 = Euclidean{128,Float64}
    U8 = Euclidean{128,UInt8}

    @test simd_type(F32, F32) == SIMD.Vec{16, Float32}
    @inferred simd_type(F32, F32)

    @test simd_type(F64, F64) == SIMD.Vec{8, Float64}
    @inferred simd_type(F64, F64)

    @test simd_type(F64, F32) == SIMD.Vec{8, Float64}
    @inferred simd_type(F64, F32)

    @test simd_type(F32, U8) == SIMD.Vec{16, Float32}
    @inferred simd_type(F32, U8)

    @test simd_type(U8, F32) == SIMD.Vec{16, Float32}
    @inferred simd_type(U8, F32)

    @test simd_type(U8, U8) == SIMD.Vec{32, Int16}
    @inferred simd_type(U8, U8)

    # Now - test that the shorter vectors also work.
    @test simd_type(Euclidean{4, Float32}, Euclidean{4, Float32}) == SIMD.Vec{4, Float32}
    @inferred simd_type(Euclidean{4, Float32}, Euclidean{4, Float32})

    # Neighbor Types
    @test GraphANN.costtype(F32, U8) == Float32
    @test GraphANN.costtype(U8, U8) == Int32

    # # Does LazyWrap also work?
    # a, b, c, d = ntuple(_ -> ntuple(_ -> rand(UInt8), 32), 4)

    # x = Euclidean((a..., b..., c..., d...,))
    # @test isa(x, Euclidean{128,UInt8})

    # sx = GraphANN._Points.LazyWrap{SIMD.Vec{32,Float32}}(x)
    # @test isa(sx, GraphANN._Points.LazyWrap)
    # @test length(sx) == div(128, 32)
    # @test length(sx[1]) == 32
    # @test eltype(sx[1]) == Float32

    # # Elements should remain equal after wrapping.
    # @test Tuple(sx[1]) == a
    # @test Tuple(sx[2]) == b
    # @test Tuple(sx[3]) == c
    # @test Tuple(sx[4]) == d
end

@testset "Testing Euclidean Calculations" begin
    unwrap = GraphANN._Points.unwrap

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

        @test isapprox(GraphANN.distance(a, b), euclidean_reference(unwrap(a), unwrap(b)))
    end

    for i in 1:50000
        a = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
        b = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))

        @test isapprox(GraphANN.distance(a, b), euclidean_reference(unwrap(a), unwrap(b)))
    end

    # Mixed types
    for i in 1:50000
        a = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
        b = GraphANN.Euclidean(ntuple(_ -> scale * rand(Float32), 128))

        @test isapprox(GraphANN.distance(a, b), euclidean_reference(unwrap(a), unwrap(b)))
    end

    # Test some unary operations.
    a = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
    b = a / 10
    @test unwrap(b) == unwrap(a) ./ 10

    c = GraphANN.Euclidean(ntuple(_ -> rand(UInt8), 128))
    @test unwrap(a + c) == unwrap(a) .+ unwrap(c)
end

