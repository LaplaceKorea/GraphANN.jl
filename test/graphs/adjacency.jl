#####
##### Default Adjacency
#####

@testset "Testing Default Adjacency" begin
    adj = [
        [2,3],
        [1],
        [2],
    ]
    x = GraphANN.DefaultAdjacencyList(adj)

    # length
    @test length(x) == 3
    @test length(x, 1) == 2
    @test length(x, 2) == 1
    @test length(x, 3) == 1

    # getindex
    @test x[1] == [2,3]
    @test x[2] == [1]
    @test x[3] == [2]

    for i in 1:length(x)
        @test GraphANN._Graphs.caninsert(x, i)
    end

    # Iteration
    @test collect(x) == adj

    # Push
    push!(x, [1,2,3])
    @test length(x) == 4
    @test x[4] == [1,2,3]

    # Insertion
    GraphANN._Graphs.unsafe_insert!(x, 1, 3, 4)
    @test x[1] == [2,3,4]

    GraphANN._Graphs.unsafe_insert!(x, 3, 1, 1)
    @test x[3] == [1,2]

    # Empty
    empty!(x, 3)
    @test x[3] == []
end

#####
##### Flat Adjacency
#####

@testset "Testing Flat Adjacency" begin
    x = GraphANN.FlatAdjacencyList{UInt32}(3, 3)

    # initial lengths
    @test GraphANN._Graphs._max_degree(x) == 3
    @test length(x) == 3
    for i in 1:length(x)
        @test length(x, i) == 0
    end

    # error throwing
    @test_throws Exception push!(x, [1,2,3])

    # initial getindex
    for i in 1:length(x)
        @test x[i] == []
    end

    #####
    ##### Inserting
    #####

    caninsert = GraphANN._Graphs.caninsert
    unsafe_insert! = GraphANN._Graphs.unsafe_insert!

    @test caninsert(x, 1)
    unsafe_insert!(x, 1, 1, 3)
    @test length(x, 1) == 1
    @test x[1] == [3]

    # insert at front
    @test caninsert(x, 1)
    unsafe_insert!(x, 1, 1, 1)
    @test length(x, 1) == 2
    @test x[1] == [1,3]

    # insert at middle
    @test caninsert(x, 1)
    unsafe_insert!(x, 1, 2, 2)
    @test length(x, 1) == 3
    @test x[1] == [1,2,3]

    # now, we should no longer be able to insert
    @test caninsert(x, 1) == false

    # iterations
    @test collect(x) == [[1,2,3], [], []]

    empty!(x, 1)
    @test length(x, 1) == 0
    @test x[1] == []

    # Construct an insert that will move two elements
    unsafe_insert!(x, 2, 1, 2)
    unsafe_insert!(x, 2, 2, 3)
    @test x[2] == [2,3]
    @test caninsert(x, 2)

    unsafe_insert!(x, 2, 1, 1)
    @test caninsert(x, 2) == false
    @test x[2] == [1,2,3]

    # Try a sorted copy
    # Here, make sure that we don't copy too many items.
    # If it DID overflow, then the trailing `4` would leak into `x[2]`.
    A = UInt32.([1,2,3,4])
    copyto!(x, 1, A)
    @test x[1] == [1,2,3]
    @test x[2] == [1,2,3]

    B = UInt32.([1,2])
    copyto!(x, 3, B)
    @test collect(x) == [[1,2,3], [1,2,3], [1,2]]
end

#####
##### Super Flat Adjacency
#####

@testset "Testing Super Flat Adjacency" begin
    x = GraphANN.SuperFlatAdjacencyList{UInt32}(3, 3)

    # initial lengths
    @test GraphANN._Graphs._max_degree(x) == 3
    @test length(x) == 3
    for i in 1:length(x)
        @test length(x, i) == 0
    end

    # error throwing
    @test_throws Exception push!(x, [1,2,3])

    # initial getindex
    for i in 1:length(x)
        @test x[i] == []
    end

    #####
    ##### Inserting
    #####

    caninsert = GraphANN._Graphs.caninsert
    unsafe_insert! = GraphANN._Graphs.unsafe_insert!

    @test caninsert(x, 1)
    unsafe_insert!(x, 1, 1, 3)
    @test length(x, 1) == 1
    @test x[1] == [3]

    # insert at front
    @test caninsert(x, 1)
    unsafe_insert!(x, 1, 1, 1)
    @test length(x, 1) == 2
    @test x[1] == [1,3]

    # insert at middle
    @test caninsert(x, 1)
    unsafe_insert!(x, 1, 2, 2)
    @test length(x, 1) == 3
    @test x[1] == [1,2,3]

    # now, we should no longer be able to insert
    @test caninsert(x, 1) == false

    # iterations
    @test collect(x) == [[1,2,3], [], []]

    empty!(x, 1)
    @test length(x, 1) == 0
    @test x[1] == []

    # Construct an insert that will move two elements
    unsafe_insert!(x, 2, 1, 2)
    unsafe_insert!(x, 2, 2, 3)
    @test x[2] == [2,3]
    @test caninsert(x, 2)

    unsafe_insert!(x, 2, 1, 1)
    @test caninsert(x, 2) == false
    @test x[2] == [1,2,3]

    # Try a sorted copy
    # Here, make sure that we don't copy too many items.
    # If it DID overflow, then the trailing `4` would leak into `x[2]`.
    A = UInt32.([1,2,3,4])
    copyto!(x, 1, A)
    @test x[1] == [1,2,3]
    @test x[2] == [1,2,3]

    B = UInt32.([1,2])
    copyto!(x, 3, B)
    @test collect(x) == [[1,2,3], [1,2,3], [1,2]]
end
