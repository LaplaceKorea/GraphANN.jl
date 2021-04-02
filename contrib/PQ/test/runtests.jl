using PQ
using Test

# main dep
using GraphANN: GraphANN

# other deps
using SIMD: SIMD
import StaticArrays: SVector

@testset "Testing PQ" begin
    @test PQ.exact_div(10, 5) == 2
    @test_throws Exception PQ.exact_div(10, 4)

    ### broadcast
    x = SVector(1, 2, 3, 4)
    y = PQ.broadcast(x)
    @test isa(y, SVector{16,Float32})
    @test y == SVector(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)

    ### sum_every
    y = SVector{16,Float32}((1:16)...)

    # 16
    x = PQ.sum_every(y, Val(16))
    @test isa(x, SVector{1,Float32})
    @test x == SVector(sum(1:16))

    # 8
    x = PQ.sum_every(y, Val(8))
    @test isa(x, SVector{2,Float32})
    @test x == SVector(sum(1:8), sum(9:16))

    # 4
    x = PQ.sum_every(y, Val(4))
    @test isa(x, SVector{4,Float32})
    @test x == SVector(sum(1:4), sum(5:8), sum(9:12), sum(13:16))

    # 2
    x = PQ.sum_every(y, Val(2))
    @test isa(x, SVector{8,Float32})
    @test x == SVector(1 + 2, 3 + 4, 5 + 6, 7 + 8, 9 + 10, 11 + 12, 13 + 14, 15 + 16)

    # 1
    x = PQ.sum_every(y, Val(1))
    @test isa(x, SVector{16,Float32})
    @test x == y

    ### Distance computations
    src = rand(SVector{4,Float32}, 256)

    # Size the destination one larger so we can detect if the last element is accidentally
    # being written to.
    dst = zeros(Float32, length(src) + 1)
    resize!(dst, length(src))
    query = rand(SVector{4,Float32})

    # This will test both the auto broadcasting as well as distance computation.
    PQ.store_distances!(dst, src, query)
    reference = GraphANN.evaluate.(Ref(GraphANN.Euclidean()), src, Ref(query))
    @test dst == reference

    # Check for overflow writes.
    resize!(dst, length(src) + 1)
    @test iszero(last(dst))
    dst[end] = 1
    resize!(dst, length(src))

    # Also try promotion.
    query = rand(SVector{4,UInt8})
    PQ.store_distances!(dst, src, query)
    reference = GraphANN.evaluate.(Ref(GraphANN.Euclidean()), src, Ref(query))
    @test dst == reference

    resize!(dst, length(src) + 1)
    @test isone(dst[end])

    ### precompute!
    queries = rand(SVector{4,Float32}, 32)
    query = SVector{sum(length, queries)}(reduce(vcat, queries))

    centroids = rand(SVector{4,Float32}, 256, 32)
    table = PQ.DistanceTable(centroids)
    PQ.precompute!(table, query)
    distances = table.distances

    for i in eachindex(queries)
        reference =
            GraphANN.evaluate.(
                Ref(GraphANN.Euclidean()), Ref(queries[i]), view(centroids, :, i)
            )
        @test reference == view(distances, :, i)
    end

    ### lookup
    queries = rand(SVector{4,Float32}, 32)
    query = SVector{sum(length, queries)}(reduce(vcat, queries))
    centroids = rand(SVector{4,Float32}, 256, 32)
    table = PQ.DistanceTable(centroids)

    # PQ method of lookup
    PQ.precompute!(table, query)
    for _ in 1:10000
        inds = ntuple(_ -> rand(UInt8), Val(length(queries)))
        dist = PQ.lookup(table, inds)

        # Manual computation.
        reference = zero(Float32)
        for i in eachindex(queries)
            reference += GraphANN.evaluate(
                GraphANN.Euclidean(),
                queries[i],
                # Add 1 to account for the `inds` being index-0.
                centroids[inds[i] + 1, i],
            )
        end
        @test isapprox(reference, dist)
    end
end # @testset
