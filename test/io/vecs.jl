@testset "Testing Vecs" begin
    # Make sure we get errors when calling the high level vecs functions.
    @test_throws ArgumentError GraphANN.load_vecs("somepath/file.qvecs")
    @test_throws ArgumentError GraphANN.load_vecs("somepath/file.jvecs")
    @test_throws ArgumentError GraphANN.load_vecs("somepath/file.avecs")
end
