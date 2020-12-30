@testset "Testing SIMDLanes" begin
    data = GraphANN.load_vecs(GraphANN.Euclidean{128,Float32}, dataset_path)
    lanes = GraphANN.SIMDLanes(SIMD.Vec{16,Float32}, data)
    @test size(lanes) == (8, length(data))

    for i in eachindex(data)
        # The resulting tuple `dd` from `cast` should have 8 `Vec{16,Float32}`
        # elements.
        dd = GraphANN._Points.cast(SIMD.Vec{16,Float32}, data[i])
        @test length(dd) == 8
        for j in 1:length(dd)
            # Use `===` because `==` defaults to elementwise comparison.
            @test dd[j] === lanes[j,i]
        end
    end

    # bounds checking
    @test_throws BoundsError lanes[0, 1]
    @test_throws BoundsError lanes[17, 1]
    @test_throws BoundsError lanes[1, 0]
    @test_throws BoundsError lanes[1, length(data) + 1]
end
