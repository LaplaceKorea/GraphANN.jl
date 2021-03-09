# Fundamentals

## Data Representation

Internal datasets are represented as [AbstractArrays](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array) whose elements are [StaticVectors](https://github.com/JuliaArrays/StaticArrays.jl).
Static Vectors are convenient and efficient for general data manipulation.
Some examples are shown below
```jldoctest
julia> using GraphANN

julia> a = GraphANN.SVector{3,UInt8}((1,2,3))
3-element StaticArrays.SVector{3, UInt8} with indices SOneTo(3):
 0x01
 0x02
 0x03

julia> b = GraphANN.SVector{3,Float32}((4,5,6))
3-element StaticArrays.SVector{3, Float32} with indices SOneTo(3):
 4.0
 5.0
 6.0

julia> a + b
3-element StaticArrays.SVector{3, Float32} with indices SOneTo(3):
 5.0
 7.0
 9.0
```
As you can see, convenient functions like addition, broadcasting, automatic-promotion etc. all work for `SVectors`.
In fact, they behave just like normal `Vectors`, but since their length is known at compile time, code generated to work on these arrays can be more efficient.
A `Vector{<:StaticArray}` (i.e., a vector with static array elements) will be densely packed, just as if the data had been stored as a dense 2D matrix.

Furthermore, since the requirement for the dataset is simply the `AbstractVector` interface, this opens up implementation possibilities.
The default is to use a standard Julia `Vector` (or a memory mapped vector when PM is being used).
However, one could imagine implementations using some kind of blocked strategy to efficiently allow insertion and deletion of elements.

## Metrics

Distance computations are performed by calling [`GraphANN.evaluate`](@ref)
```@docs
GraphANN.evaluate
```
In general, high level functions will take a `metric` keyword argument, which is then forwarded to all calls to [`GraphANN.evaluate`](@ref) internally.
Defining new metrics is simple as show below.
```jldoctest
julia> using GraphANN

julia> struct Sum end;

julia> GraphANN.evaluate(::Sum, x, y) = sum(x + y);

julia> a = GraphANN.SVector((1,2,3));

julia> GraphANN.evaluate(Sum(), a, a)
12
```

### Square Euclidean
The default metric is [`GraphANN.Euclidean`](@ref GraphANN._Base.Euclidean).
Example usage is shown below.
```jldoctest
julia> using GraphANN

julia> a = GraphANN.SVector{4,UInt8}(1, 2, 3, 4);

julia> b = GraphANN.SVector{4,Float32}(5, 6, 7, 8);

julia> GraphANN.evaluate(GraphANN.Euclidean(), a, a) # Euclidean distance between two similar data types is supported.
0

julia> GraphANN.evaluate(GraphANN.Euclidean(), b, b)
0.0f0

julia> GraphANN.evaluate(GraphANN.Euclidean(), a, b) # Mixed types is also allowed.
64.0f0
```
Internally, efficient implementations are used to compute the squared Euclidean distance.
When `UInt8` or `Int16` data types are involved, [VNNI](https://en.wikichip.org/wiki/x86/avx512_vnni) will be used as is demonstrated below.
```julia
julia> using GraphANN

julia> code_native(GraphANN.evaluate, Tuple{GraphANN.Euclidean, GraphANN.SVector{32,UInt8}, GraphANN.SVector{32,UInt8}}; syntax=:intel, debuginfo=:none)
        .text
        # Expand UInt8 to Int16
        vpmovzxbw       zmm0, ymmword ptr [rdi]
        vpmovzxbw       zmm1, ymmword ptr [rsi]
        # Subtraction
        vpsubw  zmm0, zmm0, zmm1
        vpxor   xmm1, xmm1, xmm1
        # Square Accumulate
        vpdpwssd        zmm1, zmm0, zmm0
        # Horizontal Reducing sum
        vextracti64x4   ymm0, zmm1, 1
        vpaddd  zmm0, zmm1, zmm0
        vextracti128    xmm1, ymm0, 1
        vpaddd  xmm0, xmm0, xmm1
        vpshufd xmm1, xmm0, 78                  # xmm1 = xmm0[2,3,0,1]
        vpaddd  xmm0, xmm0, xmm1
        vpshufd xmm1, xmm0, 229                 # xmm1 = xmm0[1,1,2,3]
        vpaddd  xmm0, xmm0, xmm1
        vmovd   eax, xmm0
        vzeroupper
        ret
        nop     dword ptr [rax]

```
Alternative implementations for testing are in `/etc/Alternatives`.

```@docs
GraphANN.Euclidean
```

### `costtype`

To support pre-allocation of data structures, we need to know what the type of the result from `evaluate` will be.
This is facilitated by the [`GraphANN.costtype`](@ref GraphANN.costtype) function.
```@docs
GraphANN.costtype
```

## Persistent Memory and Allocators

Constructors for many data structures take an `allocator` keyword argument.
An allocator must have the function signature
```
allocator(::Type{T}, dims...) -> AbstractArray{T,N}
```
where `N = length(dims)`.
The default allocator is [`GraphANN.stdallocator`](@ref) which simply call's Julia's normal `Array` constructor.
The other alternative is [`GraphANN.pmallocator`](@ref) which wraps a standard Julia `Array` around a pointer to persistent memory.

```@docs
GraphANN.stdallocator
GraphANN.pmallocator
```

!!! note

    At the moment, the `AbstractArray` returned by the allocator must actually be a native Julia `Array`.
    If needed, this can be fixed.

## Executors

Executors are essentially functions masquerading as loop constructs that allow portions of code to run either on a single thread or on multiple threads, depending on the executor.
Many functions throughout the GraphANN codebase will take an `executor` as an optional keyword argument to provide control over loop execution.

In general, executors have the following signature:
```julia
executor(f, [threadpool], domain, [blocksize])
```
where `f` is the function body to execute and `domain` is an indexable iterators that we want to apply `f` to elementwise.
In single-threaded execution, arguments `threadpool` and `blocksize` are ignored.
In the multi-threaded case, `threadpool` defaults to [`GraphANN.allthreads()`](@ref).
Implemented executors are listed below.
```@docs
GraphANN.single_thread
GraphANN.dynamic_thread
```

## Threading Utilities

```@docs
GraphANN._Base.ThreadPool
GraphANN._Base.allthreads
GraphANN._Base.TaskHandle
GraphANN._Base.on_threads
GraphANN._Base.ThreadLocal
```

