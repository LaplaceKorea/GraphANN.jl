@testset "DiskANN IO" begin
    #####
    ##### Graph loading and Saving
    #####

    data = GraphANN.sample_dataset()
    graph = GraphANN.load_graph(GraphANN.DiskANN(), diskann_index, length(data))
    index = GraphANN.DiskANNIndex(graph, data)

    path = tempname(@__DIR__; cleanup = true)
    GraphANN.save_graph(path, index)

    # Check that file contents match
    # NOTE: This process will not work on an arbitrary DiskANN graph since their adjacency
    # lists are unordered.
    #
    # In the Julia code, the neighbors for each vertex are ordered from smallest to largest
    # in order to match the LightGraphs API.
    #
    # To bootstrap this test, the index included with this repo is a known good sample with
    # the neighbors sorted.
    crc_original = open(crc32c, diskann_index)
    crc_new = open(crc32c, path)
    @test crc_original == crc_new

    #####
    ##### save_bin
    #####

    path = tempname(@__DIR__; cleanup = true)
    queries = GraphANN.load_vecs(Float32, query_path)
    GraphANN.save_bin(GraphANN.DiskANN(), path, queries)

    crc_original = open(crc32c, diskann_query_bin)
    crc_new = open(crc32c, path)
    @test crc_original == crc_new
end
