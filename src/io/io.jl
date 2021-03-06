module _IO

# Types
export DiskANN, SPTAG

# Functions
export load_graph, load_bin
export load_vecs, save_vecs, save_as_superflat

# local deps
import .._Base: MetaGraph, stdallocator, medioid, dynamic_thread, batched
import .._Graphs: _Graphs, UniDirectedGraph
import .._Graphs: AbstractAdjacencyList, DefaultAdjacencyList, DenseAdjacencyList
import .._Graphs: AbstractFlatAdjacencyList, FlatAdjacencyList, SuperFlatAdjacencyList
import .._Trees: TreeNode, Tree, rootindices
import ..Algorithms: DiskANNIndex

# stdlib
using Mmap: Mmap

# deps
using LightGraphs: LightGraphs
using ProgressMeter: ProgressMeter
import StaticArrays: SVector
import UnPack: @unpack

# Generic entry point for opening files.
abstract type AbstractIOFormat end
struct Native <: AbstractIOFormat end

save(path::AbstractString, args...) = save(Native(), path, args...)
function save(x::AbstractIOFormat, path::AbstractString, args...)
    return open(io -> save(x, io, args...), path; write = true)
end

export load, save, load_bin, save_bin
include("native.jl")
include("extra.jl")

# Support for DiskANN generated binary files.
include("diskann.jl")
include("sptag.jl")

# Support for the standard "*.[b|i|c]vecs" formats.
include("vecs.jl")

end # module

