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

function Base.rand(::Type{GraphANN.Euclidean{N,T}}) where {N,T}
    return GraphANN.Euclidean(ntuple(_ -> rand(T), Val(N)))
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
    @test x == x
    @test transpose(x) === x

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

    # Now - time to test that EagerWrap is doing its job.
    # The strategy is this:
    #
    # 1. Construct some tuples of UInt8
    # 2. Splat the tuples together to create a Euclidean
    # 3. SIMD wrap promoting to Float32
    # 4. Compare equality.
    a, b, c, d = ntuple(_ -> ntuple(_ -> rand(UInt8), 32), 4)

    x = GraphANN.Euclidean((a..., b..., c..., d...,))
    @test isa(x, GraphANN.Euclidean{128,UInt8})

    sx = GraphANN._Points.EagerWrap{SIMD.Vec{32,Float32}}(x)
    @test isa(sx, GraphANN._Points.EagerWrap)
    @test length(sx) == div(128, 32)
    @test length(sx[1]) == 32
    @test eltype(sx[1]) == Float32

    # Elements should remain equal after wrapping.
    @test Tuple(sx[1]) == a
    @test Tuple(sx[2]) == b
    @test Tuple(sx[3]) == c
    @test Tuple(sx[4]) == d

    # Does LazyWrap also work?
    a, b, c, d = ntuple(_ -> ntuple(_ -> rand(UInt8), 32), 4)

    x = GraphANN.Euclidean((a..., b..., c..., d...,))
    @test isa(x, GraphANN.Euclidean{128,UInt8})

    sx = GraphANN._Points.LazyWrap{SIMD.Vec{32,Float32}}(x)
    @test isa(sx, GraphANN._Points.LazyWrap)
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

@testset "Testing Wrapper Types" begin
    @testset "Packed" begin
        Packed = GraphANN._Points.Packed

        # Packing - Float32
        for i in [1, 2, 4, 8]
            N = div(16, i)
            A = [rand(GraphANN.Euclidean{N,Float32}) for _ in 1:i]
            p = Packed(A...)
            @test typeof(p.repr) == GraphANN._Points.packed_type(eltype(A))
            @test length(p) == i
            @test transpose(p) == p

            # Test distance computation.
            B = [rand(GraphANN.Euclidean{N,Float32}) for _ in 1:i]
            q = Packed(B...)
            dpacked = GraphANN.distance(p, q)
            @test isa(dpacked, SIMD.Vec{i, Float32})
            for j in 1:i
                @test isapprox(dpacked[j], GraphANN.distance(A[j], B[j]))
            end
        end

        # Packing - UInt8
        for i in [1, 2, 4, 8, 16]
            N = div(32, i)
            A = [rand(GraphANN.Euclidean{N,UInt8}) for _ in 1:i]
            p = Packed(A...)
            @test typeof(p.repr) == GraphANN._Points.packed_type(eltype(A))
            @test length(p) == i
            @test transpose(p) == p

            # Test distance computation.
            B = [rand(GraphANN.Euclidean{N,UInt8}) for _ in 1:i]
            q = Packed(B...)
            dpacked = GraphANN.distance(p, q)
            # UInt8's get promoted to Int32.
            @test isa(dpacked, SIMD.Vec{i, Int32})
            for j in 1:i
                @test isapprox(dpacked[j], GraphANN.distance(A[j], B[j]))
            end
        end

        # Test `get` and `set!` for arrays of Packed.
        a = [rand(GraphANN.Euclidean{4,Float32}) for _ in 1:4]
        b = [rand(GraphANN.Euclidean{4,Float32}) for _ in 1:4]
        A = [Packed(a...), Packed(b...)]
        @test get(A, 1, 1) == a[1]
        @test get(A, 2, 1) == a[2]
        @test get(A, 3, 1) == a[3]
        @test get(A, 4, 1) == a[4]

        @test get(A, 1, 2) == b[1]
        @test get(A, 2, 2) == b[2]
        @test get(A, 3, 2) == b[3]
        @test get(A, 4, 2) == b[4]

        # Does setting work?
        for i in 1:4
            GraphANN._Points.set!(A, b[i], i, 1)
            GraphANN._Points.set!(A, a[i], i, 2)
        end

        @test get(A, 1, 1) == b[1]
        @test get(A, 2, 1) == b[2]
        @test get(A, 3, 1) == b[3]
        @test get(A, 4, 1) == b[4]

        @test get(A, 1, 2) == a[1]
        @test get(A, 2, 2) == a[2]
        @test get(A, 3, 2) == a[3]
        @test get(A, 4, 2) == a[4]

        # Lets try a 2D array - just to make sure the logic continues working.
        c = [rand(GraphANN.Euclidean{4,Float32}) for _ in 1:4]
        d = [rand(GraphANN.Euclidean{4,Float32}) for _ in 1:4]
        A = [
             Packed(a...) Packed(b...);
             Packed(c...) Packed(d...);
        ]

        # Are things retrieved correctly?
        for i in 1:4
            @test get(A, i, 1, 1) == a[i]
            @test get(A, i, 1, 2) == b[i]
            @test get(A, i, 2, 1) == c[i]
            @test get(A, i, 2, 2) == d[i]
        end

        # Switch things around
        for i in 1:4
            GraphANN._Points.set!(A, d[i], i, 1, 1)
            GraphANN._Points.set!(A, c[i], i, 1, 2)
            GraphANN._Points.set!(A, b[i], i, 2, 1)
            GraphANN._Points.set!(A, a[i], i, 2, 2)
        end

        # Still retrieved correctly?
        for i in 1:4
            @test get(A, i, 1, 1) == d[i]
            @test get(A, i, 1, 2) == c[i]
            @test get(A, i, 2, 1) == b[i]
            @test get(A, i, 2, 2) == a[i]
        end
    end

    @testset "LazyArrayWrap" begin
        A = [rand(GraphANN.Euclidean{4,UInt8}) for _ in 1:32]
        B = [rand(GraphANN.Euclidean{4,UInt8}) for _ in 1:32]

        ea = GraphANN.Euclidean(GraphANN._Points._merge(A...))
        eb = GraphANN.Euclidean(GraphANN._Points._merge(B...))
        @test isa(ea, GraphANN.Euclidean{128,UInt8})
        @test isa(eb, GraphANN.Euclidean{128,UInt8})

        # Make sure the sub-partitions got transferred correctly.
        lazy_ea = GraphANN._Points.LazyWrap{GraphANN.Euclidean{4,UInt8}}(ea)
        lazy_eb = GraphANN._Points.LazyWrap{GraphANN.Euclidean{4,UInt8}}(eb)
        for i in 1:32
            @test lazy_ea[i] == A[i]
            @test lazy_eb[i] == B[i]
        end

        # Now, put the two of these together in a vector and make sure the correct
        # elements are extracted.
        v = [ea, eb]
        V = GraphANN._Points.LazyArrayWrap{GraphANN.Euclidean{4,Float32}}(v)
        @test size(V) == (32, 2)
        @test isa(parent(V), Vector{GraphANN.Euclidean{128,UInt8}})
        @test parent(V)[1] == ea
        @test parent(V)[2] == eb

        for i in 1:32
            @test V[i,1] == convert(GraphANN.Euclidean{4,Float32}, A[i])
            @test V[i,2] == convert(GraphANN.Euclidean{4,Float32}, B[i])
        end
    end
end
