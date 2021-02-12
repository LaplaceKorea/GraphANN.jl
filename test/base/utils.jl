@testset "Testing Utils" begin
    #-- safe_maximum
    x = [1,2,3,-1]
    @test GraphANN._Base.safe_maximum(identity, x) == 3
    @test GraphANN._Base.safe_maximum(i -> 2 * i, x) == 6
    @test GraphANN._Base.safe_maximum(identity, (i^2 for i in 1:2)) == 4
    @test GraphANN._Base.safe_maximum(identity, (i^2 for i in 1:0), 10) == 10
    @test GraphANN._Base.safe_maximum(identity, [], "hello world") == "hello world"

    #-- donothing
    @test GraphANN._Base.donothing() == nothing
    @test GraphANN._Base.donothing(1) == nothing
    @test GraphANN._Base.donothing(1, 2) == nothing
    @test GraphANN._Base.donothing(1, 2, 3) == nothing

    #-- zero! and typemax!
    x = rand(Int64, 10)
    @test !all(iszero, x)
    GraphANN._Base.zero!(x)
    @test all(iszero, x)

    GraphANN._Base.typemax!(x)
    @test all(i -> i == typemax(Int64), x)

    #-- ceiling division
    cdiv = GraphANN._Base.cdiv
    @test cdiv(10, 5) == 2
    @test cdiv(10, 4) == 3
    @test cdiv(10, 6) == 2

    # promotion
    @test cdiv(Int64(10), UInt8(6)) === Int64(2)

    #-- toeltype
    x = [rand(SVector{128,UInt8}) for _ in 1:10]
    @test eltype(x) == SVector{128,UInt8}
    y = GraphANN._Base.toeltype(Float32, x)
    @test eltype(y) == SVector{128,Float32}
    for (i, j) in zip(x, y)
        # Equal - not egal
        @test i == j
        @test i !== j
    end
end

@testset "Testing Neighbor" begin
    Neighbor = GraphANN.Neighbor
    getid = GraphANN.getid
    getdistance = GraphANN.getdistance

    # -- constructors
    x = Neighbor{Int64,Float64}()
    @test iszero(getid(x))
    @test getdistance(x) == typemax(Float64)

    # Automatically apply distance parameters.
    @test isa(Neighbor{Int64}(10, 5.0), Neighbor{Int64, Float64})
    @test isa(Neighbor{Int64}(UInt32(10), 5.0), Neighbor{Int64, Float64})

    # Full constructor.
    x = Neighbor{UInt32,Int32}(10, Int32(-1))
    @test getid(x) === UInt32(10)
    @test getdistance(x) === Int32(-1)

    # No construction without at least the `id` type parameter.
    @test_throws MethodError Neighbor(1, 1.0)
    # No automatic conversion of the distance type.
    @test_throws MethodError Neighbor{Int64,Int32}(10, Float32(5.0))

    @test GraphANN.idtype(10) == Int64
    @test GraphANN.idtype(UInt32) == UInt32
    @test GraphANN.idtype(UInt32(100)) == UInt32

    @test GraphANN.costtype(Float32) == Float32
    @test GraphANN.costtype(Float64(1.0)) == Float64
    @test GraphANN.costtype(UInt8, Int16) == Int16
    @test GraphANN.costtype(Int64(10), Float32(100)) == Float32

    # convenience wrapper for arrays
    x = [1,2,3]
    @test isa(x, Vector{Int64})
    @test GraphANN.costtype(x) == Int64

    a = Neighbor{Int64}(1, 1.0)
    b = Neighbor{Int64}(1, 2.0)
    @test GraphANN.getid(a) == 1
    @test GraphANN.getid(b) == 1
    @test GraphANN.getdistance(a) == 1.0
    @test GraphANN.getdistance(b) == 2.0

    @test getid(a) == getid(b)
    c = Neighbor{Int64}(2, 5.0)
    @test getid(a) != getid(c)

    # Ordering
    @test Neighbor{Int64}(1, 1.0) < Neighbor{Int64}(2, 2.0)
    @test Neighbor{Int64}(10, 5.0) > Neighbor{Int64}(40, 1.2)

    # Total Ordering
    @test Neighbor{Int64}(1, 1.0) < Neighbor{Int64}(2, 1.0)
    @test Neighbor{Int64}(2, 1.0) > Neighbor{Int64}(1, 1.0)

    n = Neighbor{Int64,Float32}()
    @test iszero(GraphANN.getid(n))
    @test GraphANN.getdistance(n) == typemax(Float32)

    # Array indexing.
    x = [10, 20, 30]
    i = Neighbor{Int64,Int64}(1, 1)
    @test x[i] == 10
    i = Neighbor{Int64,Int64}(2, 1)
    @test x[i] == 20
end

@testset "Testing RobinSet" begin
    x = GraphANN.RobinSet{Int}()
    @test length(x) == 0
    push!(x, 10)
    @test length(x) == 1
    push!(x, 20)
    @test length(x) == 2

    @test in(10, x) == true
    @test in(30, x) == false
    @test in(20, x) == true
    @test in(0, x) == false

    # iterator
    @test sort(collect(x)) == [10, 20]

    # deletion
    delete!(x, 10)
    @test length(x) == 1
    @test in(20, x) == true
    @test in(10, x) == false

    i = pop!(x)
    @test length(x) == 0

    push!(x, 1)
    push!(x, 2)
    @test length(x) == 2
    empty!(x)
    @test length(x) == 0
    @test in(1, x) == false
    @test in(2, x) == false
end

@testset "Testing Nearest Neighbor" begin
    # Strategy - generate a bunch of random vectors.
    # Then, make one vector very close to the query with only a small perturbation.
    # This modified vector should be the nearest neighbor.
    for i in 1:100
        x = 1000 .* rand(SVector{128, Float32}, 1000)
        y = 10 .* rand(SVector{128, Float32})
        index = rand(1:length(x))
        x[index] = y .+ (0.1 .* rand(SVector{128, Float32}))
        result = GraphANN.nearest_neighbor(y, x; metric = GraphANN.Euclidean())
        @test isa(result, NamedTuple{(:min_ind, :min_dist)})
        @test result.min_ind == index
    end

    # Basically, generate many SVectors with sequentially increasing values.
    # Then, it's pretty easy to know what the medioid will be.
    x = [@SVector fill(i, 32) for i in 1:101]

    # The average of the values from 1 to 101 is 51.
    # By construction, this will be the medioid of the dataset.
    @test all(isequal(51), x[51])
    ind = GraphANN.medioid(x)
    @test ind == 51
end

@testset "Testing Prefetch" begin
    x = [1,2,3]
    # First, just make sure this prefetch instruction exists.
    @test GraphANN.prefetch(pointer(x, 1)) == nothing
    @test GraphANN.prefetch_llc(pointer(x, 1)) == nothing

    x = [zero(SVector{32,Float32}) for _ in 1:10]
    @test GraphANN.prefetch(x, 1) == nothing
    @test GraphANN.prefetch(x, 10) == nothing

    # Supply llc prefetch as well
    @test GraphANN.prefetch(x, 5, GraphANN.prefetch_llc) == nothing
end

@testset "Testing BatchedRange" begin
    range = 1:100
    x = GraphANN.BatchedRange(range, 10)
    @test length(x) == 10
    @test x[1] == 1:10
    @test x[2] == 11:20
    @test x[10] == 91:100

    @test_throws BoundsError x[0]
    @test_throws BoundsError x[11]

    # Make sure the the last batch is handled correctly.
    range = 1:10
    x = GraphANN.BatchedRange(range, 3)
    @test x[1] == 1:3
    @test x[2] == 4:6
    @test x[3] == 7:9
    @test x[4] == 10:10
    # iteration
    @test collect(x) == [1:3, 4:6, 7:9, 10:10]

    # Finally, make sure affine translations still work.
    range = 10:2:30
    x = GraphANN.BatchedRange(range, 4)
    @test x[1] == 10:2:16
    @test x[2] == 18:2:24
    @test x[3] == 26:2:30

    # Test iteration
    @test collect(x) == [10:2:16, 18:2:24, 26:2:30]
end

# NOTE: A lot of these tests are influenced by the heap tests in DataStructures.jl
@testset "Testing Bounded Heap" begin
    vs = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    vs2 = collect(enumerate(vs))
    ordering = Base.Order.By(last)

    @testset "Constructing Heap" begin
        x = GraphANN.BoundedHeap{Int64}(Base.ForwardOrdering(), 10)
        @test length(x) == 0
        @test !GraphANN._Base.isfull(x)
        @test GraphANN._Base.getbound(x) == 10

        x = GraphANN.BoundedHeap{Int64}(Base.ReverseOrdering(), 10)
        @test length(x) == 0
        @test !GraphANN._Base.isfull(x)
        @test GraphANN._Base.getbound(x) == 10
    end

    @testset "Type Aliases" begin
        @test isa(GraphANN.BoundedMaxHeap{Int64}(10), GraphANN.BoundedMaxHeap{Int64})
        @test isa(GraphANN.BoundedMinHeap{Int64}(10), GraphANN.BoundedMinHeap{Int64})
    end

    @testset "Test BoundedMinHeap" begin
        x = GraphANN.BoundedMinHeap{Int64}(5)
        @test length(x) == 0
        @test GraphANN._Base.getbound(x) == 5
        for v in vs
            push!(x, v)
        end

        # Only keep the 5 biggest items items.
        @test length(x) == 5
        @test !isempty(x)

        # Destructive extract does not drain the heap, but it does invalidate the heap.
        items = GraphANN._Base.destructive_extract!(x)
        @test length(items) == 5
        @test length(x) == 5
        @test items == view(sort(vs; rev = true), 1:5)
        empty!(x)
        @test length(x) == 0
    end

    @testset "Test BoundedMaxHeap" begin
        x = GraphANN.BoundedMaxHeap{Int64}(5)
        @test length(x) == 0
        @test GraphANN._Base.getbound(x) == 5
        for v in vs
            push!(x, v)
        end

        # Only keep the 5 SMALLEST items.
        @test length(x) == 5
        @test !isempty(x)

        # Destructive extract does not drain the heap, but it does invalidate the heap.
        items = GraphANN._Base.destructive_extract!(x)
        @test length(items) == 5
        @test length(x) == 5
        @test items == view(sort(vs), 1:5)
        empty!(x)
        @test length(x) == 0
    end

    @testset "Test Custom Ordering" begin
        x = GraphANN.BoundedHeap{Tuple{Int,Int}}(ordering, 5)
        for v in vs2
            push!(x, v)
        end
        @test length(x) == 5
        items = GraphANN._Base.destructive_extract!(x)
        @test items == view(sort(vs2; rev = true, order = ordering), 1:5)
    end
end

@testset "Testing meanvar" begin
    for T in (Float32, UInt8), i in 1:100, corrected in (true, false)
        x = rand(T, 100)
        m, v = GraphANN._Base.meanvar(x, corrected)
        @test isapprox(m, Statistics.mean(x))
        @test isapprox(v, Statistics.var(x; corrected = corrected))
    end

    # Test inference works for things like generators.
    # Pass a somewhat complicated generator but provide a type-stable default.
    y = rand(1:100, 10)
    @inferred GraphANN._Base.meanvar(sin(Float32(i) ^ 2) for i in y; default = zero(Float32))

    # This should also work for vectors of StaticVectors
    samples = [@SVector rand(Float32, 16) for _ in 1:100]
    means, variances = GraphANN._Base.meanvar(samples)

    @test isa(means, SVector{16, Float32})
    @test isa(variances, SVector{16, Float32})

    # Compute mean and variances another way to check equality.
    samples_2d = reinterpret(reshape, Float32, samples)
    @test isapprox(means, Statistics.mean.(eachrow(samples_2d)))
    @test isapprox(variances, Statistics.var.(eachrow(samples_2d)))

    @inferred GraphANN._Base.meanvar(samples)
    y = rand(1:length(samples), 10)
    @inferred GraphANN._Base.meanvar(samples[i] for i in y; default = zero(eltype(samples)))
end
