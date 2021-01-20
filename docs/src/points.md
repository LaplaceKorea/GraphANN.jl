# Euclidean Points

One of the more important data structures in the similarity search pipeline is the `Euclidean{N,T}` data type.
Essentially, this is a statically sized collection of `N` points of type `T`.
Note the emphasis on the statically sized.
Because the length of the vectors are known at compile time, the compiler (aka LLVM) is able to completely unroll loops, which can generate extremely efficient kernels.

As an implementation detail, `Euclidean{N,T}` is simply a wrapper around a [SVector](https://github.com/JuliaArrays/StaticArrays.jl) and thus can take advantage of the work done in that package for staticallyh sized vectors.
But, since an `SVector` is basically a wrapper for an `NTuple`, we can swap back and forth between `Euclidean{N,T}`, `SVector{N,T}` and `NTuple{N,T}` trivially since they all have the same data layout.

!!! note

    Statically sizing vectors can lead to problems is the vector size is super large.
    Basically, LLVM might try to make any inner kernel a flat line of code rather than a loop, which leads to less code locality after a certain size.
    Thus, most of the kernels implemented in GraphANN try to make use of statically sized loops to let LLVM choose whether or not to unroll.
    In practice, statically sizing vectors works well up to 128.
    I've not really tried going past that though ...

## Example Usage

```julia
julia> using Random; Random.seed!(1234);
julia> x = rand(GraphANN.Euclidean{4,Float32})
Euclidean{4,Float32} <0.010770321, 0.30586517, 0.20819986, 0.40568388>

julia> y = one(GraphANN.Euclidean{4,Float32})
Euclidean{4,Float32} <1.0, 1.0, 1.0, 1.0>

julia> x + y
Euclidean{4,Float32} <1.0107703, 1.3058652, 1.2081999, 1.4056839>

julia> x - y
Euclidean{4,Float32} <-0.9892297, -0.69413483, -0.79180014, -0.5943161>

julia> convert(GraphANN.Euclidean{4, Float64}, x)
Euclidean{4,Float64} <0.010770320892333984, 0.30586516857147217, 0.2081998586654663, 0.40568387508392334>

# Return the square Euclidean distance between `x` and `y`.
julia> GraphANN.distance(x, y)
2.4405577f0
```

## Distance Computation Pipeline

Distance computation travels through a promotion pipeline to dispatch to the most efficient distance computation for the Euclidean types.
To motivate this design decision, recent AVX-512 extensions implement the VNNI class of instructions, which allows for very quick distance computation for `Int8/UInt8/Int16` types.
Furthermore, we would like to allow distance computations to be performed efficiently between mixed types.
As an example.
```julia
julia> using Random; Random.seed!(123)
julia> a = rand(GraphANN.Euclidean{32,Float32});
julia> b = rand(GraphANN.Euclidean{32,Float32});
julia> c = rand(GraphANN.Euclidean{32,UInt8});
julia> d = rand(GraphANN.Euclidean{32,UInt8});

# Float32 and Float32
julia> GraphANN.distance(a, b)
4.74954f0

# Float32 and UInt8
julia> GraphANN.distance(a, c)
620026.4f0

# UInt8 and UInt8
julia> GraphANN.distance(c, d)
284372

```
