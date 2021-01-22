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
