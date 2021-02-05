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

@testset "Testing SIMD" begin
    #####
    ##### SIMD schenangans
    #####

    _merge(x, y...) = (Tuple(x)..., _merge(y...)...)
    _merge(x) = Tuple(x)

    # Test Casting - underlying data should be preserved
    a, b, c, d = ntuple(_ -> rand(SVector{32,UInt8}), 4)
    x = SVector(_merge(a, b, c, d))

    t = GraphANN._Base.cast(SVector{32,UInt8}, x)
    @test isa(t, NTuple{4, SVector{32,UInt8}})
    @test t == (a, b, c, d)

    # For comparison with SIMD.Vec, convert both to Tuples
    t = GraphANN._Base.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, NTuple{4, SIMD.Vec{32,UInt8}})
    @test Tuple(t[1]) == Tuple(a)
    @test Tuple(t[2]) == Tuple(b)
    @test Tuple(t[3]) == Tuple(c)
    @test Tuple(t[4]) == Tuple(d)

    # Test zero padding
    a = ntuple(_ -> rand(UInt8), 30)
    x = SVector(a)
    @test isa(x, SVector{30, UInt8})
    t = GraphANN._Base.cast(SVector{32,UInt8}, x)
    @test isa(t, Tuple{SVector{32, UInt8}})
    @test Tuple(t[1])[1:30] == a
    @test iszero(t[1][31])
    @test iszero(t[1][32])

    t = GraphANN._Base.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, Tuple{SIMD.Vec{32, UInt8}})
    @test Tuple(t[1])[1:30] == a
    @test iszero(t[1][31])
    @test iszero(t[1][32])

    # Now - time to test that EagerWrap is doing its job.
    # The strategy is this:
    #
    # 1. Construct some tuples of UInt8
    # 2. Splat the tuples together to create a SVector
    # 3. SIMD wrap promoting to Float32
    # 4. Compare equality.
    a, b, c, d = ntuple(_ -> rand(SVector{32,UInt8}), 4)
    x = SVector(_merge(a, b, c, d))
    @test isa(x, SVector{128,UInt8})

    sx = GraphANN._Base.EagerWrap{SIMD.Vec{32,Float32}}(x)
    @test isa(sx, GraphANN._Base.EagerWrap)
    @test length(sx) == div(128, 32)
    @test length(sx[1]) == 32
    @test eltype(sx[1]) == Float32

    # Elements should remain equal after wrapping.
    @test Tuple(sx[1]) == Tuple(a)
    @test Tuple(sx[2]) == Tuple(b)
    @test Tuple(sx[3]) == Tuple(c)
    @test Tuple(sx[4]) == Tuple(d)

    # Promotion Logic
    find_distance_type = GraphANN._Base.find_distance_type

    # First, test the fallback
    @test find_distance_type(Float32, Int) == promote_type(Float32, Int)
    @test find_distance_type(Int, Float32) == promote_type(Float32, Int)

    # now, test our specializations for UInt8 work
    @test find_distance_type(UInt8, UInt8) == Int16
    @test find_distance_type(UInt8, Int16) == Int16
    @test find_distance_type(Int16, UInt8) == Int16

    # Manually try out some promotions - especially check for inference.
    simd_type = GraphANN._Base.simd_type

    # Full AVX-512 vector
    F32 = SVector{128,Float32}
    F64 = SVector{128,Float64}
    U8 = SVector{128,UInt8}

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
    @test simd_type(SVector{4, Float32}, SVector{4, Float32}) == SIMD.Vec{4, Float32}
    @inferred simd_type(SVector{4, Float32}, SVector{4, Float32})

    # Neighbor Types
    @test GraphANN.costtype(F32, U8) == Float32
    @test GraphANN.costtype(U8, U8) == Int32
end

@testset "Testing Euclidean Calculations" begin
    # Lets do some distance calculations
    scale = 100
    metric = GraphANN.Euclidean()
    for i in 1:50000
        a = scale .* rand(SVector{128,Float32})
        b = scale .* rand(SVector{128,Float32})

        @test isapprox(
            GraphANN.evaluate(metric, a, b),
            euclidean_reference(a, b)
        )
    end

    for i in 1:50000
        a = rand(SVector{128,UInt8})
        b = rand(SVector{128,UInt8})
        @test isapprox(
            GraphANN.evaluate(metric, a, b),
            euclidean_reference(a, b)
        )
    end

    # Mixed types
    for i in 1:50000
        a = rand(SVector{128,UInt8})
        b = scale .* rand(SVector{128,Float32})

        @test isapprox(
            GraphANN.evaluate(metric, a, b),
            euclidean_reference(a, b)
        )
    end
end

