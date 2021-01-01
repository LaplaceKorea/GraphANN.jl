_get(x::AbstractVector, inds::Tuple) = getindex.((x,), inds)
_merge(x, y...) = (Tuple(x)..., _merge(y...)...)
_merge(x) = (Tuple(x)...,)

function encoding_test(encoder, vectors::AbstractVector{SIMD.Vec{N,T}}) where {N,T}
    n = length(vectors)
    iter = Iterators.product(1:n, 1:n, 1:n, 1:n)

    for tup in iter
        query = GraphANN.Euclidean(_merge(_get(vectors, tup)...))
        @test isa(query, GraphANN.Euclidean{4 * N,T})

        code = GraphANN.encode(UInt8, encoder, query)
        @test isa(code, NTuple{4,UInt8})
        @test code == (tup .- 1)
    end

    # Now - does the bulk encoding work?
    queries = [GraphANN.Euclidean(_merge(_get(vectors, i)...)) for i in iter] |> vec
    encoded = GraphANN.encode(UInt32, encoder, queries)
    for (code, gt) in zip(encoded, iter)
        @test code == gt .- 1
    end
end

function float32_base()
    return map(1:8) do i
        v = zeros(Float32, 8)
        # Slight perturbation
        v[i] = (4 * i) + randn(Float32)

        # Set two values not at index `i` to `-i`.
        while true
            others = rand(1:8, 2)
            if !in(i, others)
                setindex!.(Ref(v), -i, others)
                break
            end
        end
        return SIMD.Vec(v...)
    end
end

function uint8_base()
    return map(1:8) do i
        v = zeros(UInt8, 8)
        # Slight perturbation
        v[i] = (4 * i)

        # Set two values not at index `i` to `1`.
        # Can't set negative like the Float32 case because ... these are unsigned
        # integers.
        while true
            others = rand(1:8, 2)
            if !in(i, others)
                setindex!.(Ref(v), i, others)
                break
            end
        end
        return SIMD.Vec(v...)
    end
end

@testset "Testing PQ" begin
    # Strategy - manually build a PQ table with some simple points in it.
    # Make sure everything does, well, what it's supposed to!

    # First - manually construct some known centroids, setting values on the diagonal
    # and off diagonal to add a little bit of noise.
    flat_centroids = map(1:8) do i
        v = zeros(Float32, 8)
        v[1] = 1
        v[i] = 4 * i
        return SIMD.Vec(v...)
    end
    @test isa(flat_centroids, Vector{SIMD.Vec{8,Float32}})

    # Create a 4-dimensional partition so we're sure to cover more than one cache line.
    centroids = reduce(hcat, ntuple(_ -> GraphANN.Euclidean.(flat_centroids), 4))
    pqtable = GraphANN.PQTable{size(centroids, 2)}(centroids)

    # Make sure we get an error if we try to construct a PQTable with the wrong dimension.
    @test_throws ArgumentError GraphANN.PQTable{5}(centroids)

    #####
    ##### Encoding.
    #####

    # First - create a set of 8 vectors where we know `vector[i]` is nearest
    # `flat_centroids[i]`.
    # We can then know exactly what the result is supposed to be when we construct full
    # length vectors.
    base_vectors = float32_base()
    @test isa(base_vectors, Vector{SIMD.Vec{8,Float32}})

    # Make sure we constructed the base vectors correctly.
    for i in 1:8
        v = base_vectors[i]
        distances = sum.((Ref(v) .- flat_centroids) .^ 2)
        _, indmin = findmin(distances)
        @test indmin == i
    end

    # Try all combinations!
    encoding_test(pqtable, base_vectors)

    # Next - switch over to the fast encoder, which should work for the types we have here.
    encoder = GraphANN._Quantization.binned(pqtable)
    encoding_test(encoder, base_vectors)

    # Next - what happens if we use uint8 bases
    base_vectors = uint8_base()
    @test isa(base_vectors, Vector{SIMD.Vec{8,UInt8}})

    # Make sure we constructed the base vectors correctly.
    for i in 1:8
        # In this case - we need to actually perform a conversion because SIMD.jl does
        # not implement automatic promotion.
        v = convert(SIMD.Vec{8,Float32}, base_vectors[i])
        distances = sum.((Ref(v) .- flat_centroids) .^ 2)
        _, indmin = findmin(distances)
        @test indmin == i
    end

    encoding_test(pqtable, base_vectors)

    # Next - switch over to the fast encoder, which should work for the types we have here.
    encoder = GraphANN._Quantization.binned(pqtable)
    encoding_test(encoder, base_vectors)
end
