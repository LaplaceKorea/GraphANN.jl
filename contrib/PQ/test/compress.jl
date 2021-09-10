@testset "Testing Squish" begin
    zero_slices = [1,3,5,6,8]
    matrix = randn(Float32, 10, 10)
    vector = randn(Float32, 10)
    for i in zero_slices
        view(matrix, :, i) .= 0
        view(matrix, i, :) .= 0
        vector[i] = 0
    end

    nt = PQ.squish(matrix, vector)
    @test isa(nt, NamedTuple)
    @test keys(nt) == (:matrix, :vector, :indices)

    indices = sort(setdiff(1:10, zero_slices))
    @test nt.indices == indices
    @test nt.matrix == view(matrix, indices, indices)
    @test nt.vector == view(vector, indices)
end
