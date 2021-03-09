# Saving and Loading

The GraphANN repo provides many methods for saving and loading datasets and graphs, both for the GraphANN and for other projects like DiskANN and SPTAG.
This file summarizes those methods.

## Native Formats

These are formats meant specifically for `GraphANN.jl`.

### General Binary Formats

#### Graphs

Save and load graphs in a general binary format using the
[`load`](@ref GraphANN.load(::Type{<:GraphANN._Graphs.AbstractAdjacencyList}, ::AbstractString)) and [`save`](@ref GraphANN.save(::AbstractString, ::Any)) functions.
The general binary storage format for graphs is as follows.

* **header**: 32B (4x8B integers).
    - 1st (`sizeof(T)`):  Size of the element type `T` in bytes encoding the graph.
            For `UInt32` integer storage, this will be 4. For `UInt64` storage, this
            will be 8.  A full 8B is reserved for this slot to allow for future expansion.
    - 2nd (`nv`):  Total number of vertices stored in the graph.
    - 3rd (`ne`):  Total number of edges stored in the graph.
    - 4th:  Maximum out degree of the graph.

* **Out Degrees**: What follows are `nv` number of integers each taking `sizeof(T)` bytes.
    Each integer is the positiion-wise out degree of its corresponding vertex.
    So, the first value will the the out degree for vertex 1, the second value will be the
    out degree for vertex 2 etc.

* **Adjacency Lists**: The last portion of the file is `ne` integers with size `sizeof(T)`.
    These are the out neighbors of each vertex. The graph can be reconstructed in
    combination with the `out degrees` read above to correlate regions of the densely packed
    adjacency list with the adjacency list for each vertex.

```@docs
GraphANN.load(::Type{<:GraphANN._Graphs.AbstractAdjacencyList}, ::AbstractString)
GraphANN.save(::AbstractString, ::Any)
```

!!! note

    The adjacency lists for graphs stored by GraphANN are stored using index-1 for the
    neighbors. That is, the minimal index is "1" and not "0".

### Fast Binary Formats

```@docs
GraphANN.save_bin(::AbstractString, ::AbstractVector)
GraphANN.save_bin(::AbstractString, ::GraphANN.UniDirectedGraph{T, GraphANN.DenseAdjacencyList{T}}) where {T}
GraphANN.load_bin
```

## VECS Formats

A common way to ship around data is in the [vecs](http://corpus-texmex.irisa.fr/) file format.
GraphANN has the functions [`load_vecs`](@ref GraphANN.load_vecs) and [`save_vecs`](@ref GraphANN.save_vecs) to help read and create these datasets.
```@docs
GraphANN.load_vecs
GraphANN.save_vecs
```

## DiskANN Formats

We are capable of reading and generating DiskANN compatible indexes and binary files.
To control ambiguities and allow for future potential expandability, the singleton type [`GraphANN.DiskANN`](@ref) is used as the first argument for some methods.

```@docs
GraphANN.DiskANN
GraphANN.load_graph(::GraphANN.DiskANN, ::AbstractString, ::Any)
GraphANN.save_graph(::AbstractString, ::GraphANN.DiskANNIndex)
GraphANN.save_bin(::GraphANN._IO.DiskANN, ::AbstractString, ::AbstractMatrix)
```
