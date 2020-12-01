@testset "Testing Telemetry" begin
    # Test `find`
    find = GraphANN._Telemetry.find

    tup = (1.0, "hello")
    @test find(Integer, tup...) == nothing
    @test find(AbstractFloat, tup...) == 1.0
    @test find(Float64, tup...) == 1.0
    @test find(Float32, tup...) == nothing
    @test find(String, tup...) == "hello"
    @test find(AbstractString, tup...) == "hello"

    @test find(Int64, tup..., 10) == 10

    # To test if functions are being called, we use a closure to modify `x`.
    maybecall = GraphANN._Telemetry.maybecall
    x = Ref(1)
    f(i = 1) = (x[] += i; x[])

    # Make sure the closure works as expected ...
    @test x[] == 1
    f()
    @test x[] == 2

    maybecall(f, nothing)
    @test x[] == 2

    old = x[]
    maybecall(f, 100)
    new = x[]
    @test new - old == 100

    # Now, make sure the top level telemetry is working.
    t = GraphANN.Telemetry(; a = -1, b = 1.0, c = "hello")

    # Property access
    @test isa(t.val, NamedTuple)
    @test t.a == -1
    @test t.b == 1.0
    @test t.c == "hello"

    old = x[]
    GraphANN.ifhasa(f, t, Integer)
    new = x[]
    @test new - old == t.a

    # `f` should NOT be called in this case since `t` does not contain a `Complex{Float32}`
    old = x[]
    GraphANN.ifhasa(f, t, Complex{Float32})
    new = x[]
    @test old == new

    # Make sure other positions work.
    old = x[]
    GraphANN.ifhasa(f, t, Float64)
    new = x[]
    @test new - old == t.b

    array = []
    GraphANN.ifhasa(t, String) do v
        push!(array, v)
    end
    @test array == [t.c]
end
