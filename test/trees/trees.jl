# Utility struct for counting nodes in the tree.
mutable struct NodeCounter
    count::Int
end
NodeCounter() = NodeCounter(0)
(x::NodeCounter)(_) = x.count += 1
Base.getindex(x::NodeCounter) = x.count

@testset "Testing Tree" begin
    #####
    ##### Tree Node
    #####

    TreeNode = GraphANN._Trees.TreeNode

    #-- Constructors
    # Empty constructos
    x = TreeNode{UInt32}()
    @test x.id === zero(UInt32)
    @test x.id !== zero(UInt64)
    @test x.childstart === zero(UInt32)
    @test x.childstart !== zero(UInt64)
    @test x.childend === zero(UInt32)
    @test x.childend !== zero(UInt64)
    @test GraphANN._Trees.isnull(x)
    @test GraphANN._Trees.isleaf(x)
    @test GraphANN.getid(x) == 0
    @test GraphANN._Trees.childindices(x) == 0:0
    @test TreeNode{UInt32}(x) === x

    x = TreeNode{UInt64}()
    @test x.id !== zero(UInt32)
    @test x.id === zero(UInt64)
    @test x.childstart !== zero(UInt32)
    @test x.childstart === zero(UInt64)
    @test x.childend !== zero(UInt32)
    @test x.childend === zero(UInt64)
    @test GraphANN._Trees.isnull(x)
    @test GraphANN._Trees.isleaf(x)
    @test GraphANN.getid(x) == 0
    @test GraphANN._Trees.childindices(x) == 0:0
    @test TreeNode{UInt64}(x) === x

    # Single integer constructor
    x = TreeNode{UInt32}(10)
    @test GraphANN.getid(x) == 10
    @test GraphANN._Trees.isnull(x) == false
    @test GraphANN._Trees.isleaf(x) == true

    # Comparison
    # These nodes should not be leaves since they have children.
    x = TreeNode(10, 20, 30)
    y = TreeNode(5, 30, 40)
    @test y < x

    @test GraphANN._Trees.isnull(x) == false
    @test GraphANN._Trees.isleaf(x) == false
    @test GraphANN._Trees.isnull(y) == false
    @test GraphANN._Trees.isleaf(y) == false
    @test GraphANN._Trees.childindices(x) == 20:30
    @test GraphANN._Trees.childindices(y) == 30:40

    # Scalar Broadcasting
    A = Vector{TreeNode{Int64}}(undef, 10)
    A .= x
    @test all(isequal(x), A)

    # Reading and Writing
    buffer = IOBuffer()
    write(buffer, x)
    seekstart(buffer)
    y = read(buffer, typeof(x))
    @test y === x

    #####
    ##### Tree
    #####

    Tree = GraphANN._Trees.Tree

    # Test empty constructor
    tree = Tree{UInt32}(10)
    @test isa(tree, Tree)
    @test length(GraphANN._Trees.rootindices(tree)) == 0
    @test all(GraphANN._Trees.isnull, tree.nodes)
    @test all(GraphANN._Trees.isleaf, tree.nodes)

    function test_tree()
        # Manually construct a dummy tree for testing purposes.
        # The pattern made by the node constructors describes the shape of the tree.
        nodes = [
            TreeNode{UInt32}(1, 3, 4),
            TreeNode{UInt32}(2, 5, 8),
            #=1=#   TreeNode{UInt32}(3, 9, 10),
            #=1=#   TreeNode{UInt32}(4, 0, 0),

            #=2=#   TreeNode{UInt32}(5, 0, 0),
            #=2=#   TreeNode{UInt32}(6, 0, 0),
            #=2=#   TreeNode{UInt32}(7, 0, 0),
            #=2=#   TreeNode{UInt32}(8, 11, 11),

            #=3=#       TreeNode{UInt32}(9, 0, 0),
            #=3=#       TreeNode{UInt32}(10, 0, 0),

            #=4=#       TreeNode{UInt32}(11, 0, 0),
        ]

        # Root nodes end at index 2.
        return Tree(2, nodes)
    end

    tree = test_tree()
    indices_seen = GraphANN._Trees.ispacked(tree)
    @test all(isequal(true), indices_seen)
    @test length(indices_seen) == 11

    roots = GraphANN._Trees.roots(tree)
    @test length(roots) == 2
    @test roots[1] === TreeNode{UInt32}(1, 3, 4)
    @test roots[2] === TreeNode{UInt32}(2, 5, 8)

    children = GraphANN._Trees.children(tree, roots[1])
    @test length(children) == 2
    @test children[1] == TreeNode{UInt32}(3, 9, 10)
    @test GraphANN._Trees.isleaf(children[2])
    @test children == GraphANN._Trees.children(tree, 1)

    children = GraphANN._Trees.children(tree, roots[2])
    @test length(children) == 4
    @test GraphANN._Trees.isleaf(children[1])
    @test GraphANN._Trees.isleaf(children[2])
    @test GraphANN._Trees.isleaf(children[3])

    children = GraphANN._Trees.children(tree, children[4])
    @test length(children) == 1

    # Build some visiting function to work with the `onnodes` function.
    onnodes = GraphANN._Trees.onnodes
    counter = NodeCounter()
    onnodes(counter, tree)
    @test counter[] == length(nodes)

    # repeat an index - make sure the `ispacked` function catches it.
    nodes = tree.nodes
    node = nodes[3]
    nodes[3] = TreeNode{UInt32}(node.id, node.childstart, node.childend + one(UInt32))
    @test_throws Exception GraphANN._Trees.validate(tree)

    #####
    ##### TreeBuilder
    #####

    TreeBuilder = GraphANN._Trees.TreeBuilder
    builder = TreeBuilder{Int64}(100)

    # Pull out the `nodes` vector to manually inspect during construction.
    tree = builder.tree
    nodes = builder.tree.nodes
    @test all(GraphANN._Trees.isnull, nodes)

    range = GraphANN._Trees.initnodes!(builder, 1:10)
    @test range == 1:10
    counter = NodeCounter()
    onnodes(counter, tree)
    @test counter[] == 10

    @test !any(GraphANN._Trees.isnull, view(nodes, 1:10))
    @test all(GraphANN._Trees.isnull, view(nodes, 11:lastindex(nodes)))
    @test all(GraphANN._Trees.isleaf, nodes)

    # Make sure we can't add the root again.
    @test_throws AssertionError GraphANN._Trees.initnodes!(builder, 20:30)
    range = GraphANN._Trees.addnodes!(builder, 1, 11:20)
    @test range == 11:20
    @test GraphANN._Trees.isleaf(nodes[1]) == false
    @test GraphANN._Trees.childindices(nodes[1]) == range

    # Make sure we still visit all the nodes.
    counter = NodeCounter()
    onnodes(counter, tree)
    @test counter[] == 20

    range = GraphANN._Trees.addnodes!(builder, 11, 30:40)
    @test range == 21:31
    @test GraphANN._Trees.childindices(nodes[11]) == range
    @test GraphANN._Trees.isleaf(nodes[11]) == false

    counter = NodeCounter()
    onnodes(counter, tree)
    @test counter[] == 31

    nodes_visited = Int[]
    onnodes(x -> push!(nodes_visited, GraphANN.getid(x)), tree)
    sort!(nodes_visited)
    @test nodes_visited == sort(reduce(vcat, (1:10, 11:20, 30:40)))

    # Finially, test that we can call `addnodes!` with a zero parent and still
    # get the behavior of `initnodes!`.

    builder = TreeBuilder{Int64}(100)

    # Pull out the `nodes` vector to manually inspect during construction.
    tree = builder.tree
    nodes = builder.tree.nodes
    @test all(GraphANN._Trees.isnull, nodes)

    range = GraphANN._Trees.addnodes!(builder, 0, 1:10)
    @test range == 1:10
    counter = NodeCounter()
    onnodes(counter, tree)
    @test counter[] == 10
end
