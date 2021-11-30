# Reference implementation
# This isn't exactly meant to be robust - just be pretty obcvous
function reference(::GraphANN.Euclidean, A, B)
    s = zero(Float64)
    for (a, b) in zip(A, B)
        c = Float64(a) - Float64(b)
        s += c^2
    end
    return s
end

function reference(::GraphANN.InnerProduct, A, B)
    s = zero(Float64)
    for (a, b) in zip(A, B)
        s += Float64(a) * Float64(b)
    end
    return s
end

"""
    nocalls(f, args...)

Test that the native code generated for `f(args...)` does not contain the `call` x86
instruction.

In other words - everything gets inlined.
"""
function nocalls(f, args...)
    io = IOBuffer()
    code_native(io, f, Tuple{map(typeof, args)...}; syntax = :intel, debuginfo = :none)
    seekstart(io)
    str = read(io, String)
    return !occursin("call", str)
end

"""
    nojumpa(f, args...)

Test that the native code generated for `f(args...)` does not contain any jumps.

In other words - everything gets completely unrolled.
"""
function nojumps(f, args...)
    io = IOBuffer()
    code_native(io, f, Tuple{map(typeof, args)...}; syntax = :intel, debuginfo = :none)
    seekstart(io)
    str = read(io, String)
    return !any(occursin(str), ("jmp", "jne", "je"))
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

    # For comparison with SIMD.Vec, convert both to Tuples
    t = GraphANN._Base.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, NTuple{4,SIMD.Vec{32,UInt8}})
    @test Tuple(t[1]) == Tuple(a)
    @test Tuple(t[2]) == Tuple(b)
    @test Tuple(t[3]) == Tuple(c)
    @test Tuple(t[4]) == Tuple(d)

    # Test zero padding
    a = ntuple(_ -> rand(UInt8), 30)
    x = SVector(a)
    @test isa(x, SVector{30,UInt8})

    t = GraphANN._Base.cast(SIMD.Vec{32,UInt8}, x)
    @test isa(t, Tuple{SIMD.Vec{32,UInt8}})
    @test Tuple(t[1])[1:30] == a
    @test iszero(t[1][31])
    @test iszero(t[1][32])

    # Now - time to test that ValueWrap is doing its job.
    # The strategy is this:
    #
    # 1. Construct some tuples of UInt8
    # 2. Splat the tuples together to create a SVector
    # 3. SIMD wrap promoting to Float32
    # 4. Compare equality.
    a, b, c, d = ntuple(_ -> rand(SVector{16,UInt8}), 4)
    x = SVector(_merge(a, b, c, d))
    @test isa(x, SVector{64,UInt8})

    sx = GraphANN._Base.ValueWrap{16,Float32}(x)
    @test isa(sx, GraphANN._Base.ValueWrap)
    @test length(sx) == div(64, 16)
    @test length(sx[1]) == 16
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
    @test find_distance_type(UInt8, Int8) == Int16
    @test find_distance_type(Int8, UInt8) == Int16
    @test find_distance_type(Int8, Int8) == Int16

    @test find_distance_type(UInt8, Int16) == Int16
    @test find_distance_type(Int8, Int16) == Int16
    @test find_distance_type(Int16, UInt8) == Int16
    @test find_distance_type(Int16, Int8) == Int16

    @test find_distance_type(Float16, Float16) == Float32
    @test find_distance_type(Float16, Float32) == Float32
    @test find_distance_type(Float16, UInt8) == Float32

    # Manually try out some promotions - especially check for inference.
    simd_type = GraphANN._Base.simd_type

    # Full AVX-512 vector
    F32 = SVector{128,Float32}
    F64 = SVector{128,Float64}
    U8 = SVector{128,UInt8}

    @test simd_type(F32, F32) == SIMD.Vec{16,Float32}
    @inferred simd_type(F32, F32)

    @test simd_type(F64, F64) == SIMD.Vec{8,Float64}
    @inferred simd_type(F64, F64)

    @test simd_type(F64, F32) == SIMD.Vec{8,Float64}
    @inferred simd_type(F64, F32)

    @test simd_type(F32, U8) == SIMD.Vec{16,Float32}
    @inferred simd_type(F32, U8)

    @test simd_type(U8, F32) == SIMD.Vec{16,Float32}
    @inferred simd_type(U8, F32)

    @test simd_type(U8, U8) == SIMD.Vec{32,Int16}
    @inferred simd_type(U8, U8)

    # Now - test that the shorter vectors also work.
    @test simd_type(SVector{4,Float32}, SVector{4,Float32}) == SIMD.Vec{4,Float32}
    @inferred simd_type(SVector{4,Float32}, SVector{4,Float32})

    # Neighbor Types
    E = GraphANN.Euclidean()
    @test GraphANN.costtype(E, F32, U8) == Float32
    @test GraphANN.costtype(E, U8, U8) == Int32
end

function test_metric(V::Vector; kw...)
    return test_metric(V[1], V[2], pointer(V, 1), pointer(V, 2); kw...)
end
function test_metric(A::Vector, B::Vector; kw...)
    return test_metric(A[1], B[1], pointer(A, 1), pointer(B, 1); kw...)
end

function test_metric(
    a,
    b,
    pa::Ptr,
    pb::Ptr;
    metric = Euclidean,
    test_codegen = false,
    rtol = sqrt(eps(Float32)),
)
    ref = reference(metric, a, b)
    # Try all pointer/value pairs
    @test isapprox(GraphANN.evaluate(metric, a, b), ref; rtol)
    @test isapprox(GraphANN.evaluate(metric, pa, pb), ref; rtol)
    @test isapprox(GraphANN.evaluate(metric, pa, b), ref; rtol)
    @test isapprox(GraphANN.evaluate(metric, a, pb), ref; rtol)

    # Generated Code Tests
    if test_codegen
        @test nocalls(GraphANN.evaluate, metric, a, b)
        @test nocalls(GraphANN.evaluate, metric, pa, pb)
        @test nocalls(GraphANN.evaluate, metric, pa, b)
        @test nocalls(GraphANN.evaluate, metric, a, pb)

        @test nojumps(GraphANN.evaluate, metric, a, b)
        @test nojumps(GraphANN.evaluate, metric, pa, pb)
        @test nojumps(GraphANN.evaluate, metric, pa, b)
        @test nojumps(GraphANN.evaluate, metric, a, pb)
    end

    passed = true
    passed &= isapprox(GraphANN.evaluate(metric, a, b), ref; rtol)
    passed &= isapprox(GraphANN.evaluate(metric, pa, pb), ref; rtol)
    passed &= isapprox(GraphANN.evaluate(metric, pa, b), ref; rtol)
    passed &= isapprox(GraphANN.evaluate(metric, a, pb), ref; rtol)

    if !passed
        @show repr(a) repr(b)
        error()
    end
end

@testset "Testing Euclidean Calculations" begin
    # Lets do some distance calculations
    lengths = [100, 128]
    left_types = [Float32, Float16, UInt8, Int8]
    right_types = [Float32, Float16, UInt8, Int8]
    metric = GraphANN.Euclidean()

    for (left, right, len) in Iterators.product(left_types, right_types, lengths)
        x = Vector{SVector{len,left}}(undef, 1)
        y = Vector{SVector{len,right}}(undef, 1)
        test_codegen = true
        for i in 1:2000
            x[1] = rand(SVector{len,left})
            y[1] = rand(SVector{len,right})
            test_metric(x, y; metric, test_codegen)
            test_codegen = false
        end
    end

    # Scalar broadcasting.
    x = rand(SVector{128,Float32}, 3)
    y = rand(SVector{128,Float32}, 3)

    metric = GraphANN.Euclidean()
    d = GraphANN.evaluate.(metric, x, y)
    for i in eachindex(x, y)
        @test isapprox(d[i], GraphANN.evaluate(metric, x[i], y[i]))
    end
end

maybescale(x::SVector{N,Float32}, scale) where {N} = scale .* x
maybescale(x::SVector, scale) = x

@testset "Testing InnerProduct Calculations" begin
    # Lets do some distance calculations
    lengths = [100, 128]
    left_types = [Float32, Float16, UInt8, Int8]
    right_types = [Float32, Float16, UInt8, Int8]
    scale = 100
    metric = GraphANN.InnerProduct()

    for (left, right, len) in Iterators.product(left_types, right_types, lengths)
        x = Vector{SVector{len,left}}(undef, 1)
        y = Vector{SVector{len,right}}(undef, 1)
        test_codegen = true
        for i in 1:2000
            x[1] = maybescale(rand(SVector{len,left}), scale)
            y[1] = maybescale(rand(SVector{len,right}), scale)
            test_metric(x, y; metric, test_codegen, rtol = 0.1)
            test_codegen = false
        end
    end

    # Scalar broadcasting.
    x = rand(SVector{128,Float32}, 3)
    y = rand(SVector{128,Float32}, 3)

    metric = GraphANN.InnerProduct()
    d = GraphANN.evaluate.(metric, x, y)
    for i in eachindex(x, y)
        @test isapprox(d[i], GraphANN.evaluate(metric, x[i], y[i]))
    end
end

