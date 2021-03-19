@testset "Testing SPTAG IO" begin
    @testset "Tree Reader" begin
        # import some functions to make life a little easier here.
        issentinel = GraphANN._IO.issentinel
        maybeadjust = GraphANN._IO.maybeadjust
        adjust = GraphANN._IO.adjust

        # The SPTAG code uses the "-1" value to represend null fields.
        # For example, child nodes will have both their child start and child end fields
        # set to "-1".
        #
        # On the Julia side of things, we want to be able to work with these values as
        # either signed or unsigned.
        # Thus, the Julia code uses "0" as a sentinel value, which works out super well
        # since Julia is index-1.
        @test issentinel(-1)
        @test issentinel(-Int32(1))
        @test !issentinel(typemax(Int64))
        @test !issentinel(typemax(Int32))
        @test !issentinel(0)
        @test !issentinel(Int32(0))

        @test issentinel(typemax(UInt8))
        @test issentinel(typemax(UInt16))
        @test issentinel(typemax(UInt32))
        @test issentinel(typemax(UInt64))
        @test !issentinel(zero(UInt64))

        # With these working, time to checkout the methods that adjust fields in read
        # `TreeNodes` from their C++ representation to the Julia representation.
        @test maybeadjust(10) == 10
        @test maybeadjust(10, 1) == 11
        @test maybeadjust(10, -1) == 9
        @test maybeadjust(-1) == 0
        @test maybeadjust(-1, 100) == 0
        @test maybeadjust(typemax(UInt32)) == 0
        @test maybeadjust(UInt32(10)) == 10
        @test maybeadjust(UInt32(10), 4) == 14

        # Now try adjusting TreeNodes
        TreeNode = GraphANN._Trees.TreeNode
        @test adjust(TreeNode(0, 10, 20)) == TreeNode(1, 10, 19)
        @test adjust(TreeNode(-1, -1, -1)) == TreeNode(0, 0, 0)
        @test adjust(TreeNode(10, 100, 120)) == TreeNode(11, 100, 119)

        a = TreeNode(zero(UInt32), UInt32(10), UInt32(20))
        b = TreeNode(UInt32(1), UInt32(10), UInt32(19))
        @test adjust(a) == b

        a = TreeNode(typemax(UInt32), typemax(UInt32), typemax(UInt32))
        b = TreeNode(zero(UInt32), zero(UInt32), zero(UInt32))
        @test adjust(a) == b
    end

    @testset "Tree Writes" begin
        original_path = joinpath(sptag_dir, "siftsmall", "tree.bin")
        tree = GraphANN._IO.load_bktree(original_path)

        temp_path = tempname(@__DIR__; cleanup = true)
        GraphANN.save(GraphANN.SPTAG(), temp_path, tree.tree)

        # Check that file contents match
        crc_original = open(crc32c, original_path)
        crc_new = open(crc32c, temp_path)
        @test crc_original == crc_new
    end

    @testset "Graph Round Trip" begin
        original_path = joinpath(sptag_dir, "siftsmall", "graph.bin")

        # N.B.: Like DiskANN, SPTAG doesn't keep its adjacency lists sorted by index,
        # which is expected by LightGraphs.jl
        #
        # SO, when we're performing the comparison, don't sort so we preserve the original
        # ordering.
        graph = GraphANN._IO.load_graph(GraphANN.SPTAG(), original_path; sort = false)

        temp_path = tempname(@__DIR__; cleanup = true)
        GraphANN.save(GraphANN.SPTAG(), temp_path, graph)

        # Check that file contents match
        crc_original = open(crc32c, original_path)
        crc_new = open(crc32c, temp_path)
        @test crc_original == crc_new
    end

    @testset "Testting Misc" begin
        data = GraphANN.sample_dataset()
        # Note: `mktempdir` removes the directory when done.
        mktempdir() do dir
            GraphANN._IO.generate_misc(GraphANN.SPTAG(), dir, data)

            paths = [
                "metadata.bin",
                "metadataIndex.bin",
                "deletes.bin",
            ]

            for path in paths
                crc_original = open(crc32c, joinpath(sptag_dir, "siftsmall", path))
                crc_new = open(crc32c, joinpath(dir, path))
                @test crc_original == crc_new
            end
        end
    end
end
