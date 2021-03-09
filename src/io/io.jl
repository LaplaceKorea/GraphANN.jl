module _IO

# Types
export DiskANN, SPTAG

# Functions
export load_graph, save_graph
export load_vecs, save_vecs

# local deps
import .._Base: MetaGraph, stdallocator, medioid
import .._Graphs: _Graphs, UniDirectedGraph
import .._Graphs: DefaultAdjacencyList, FlatAdjacencyList, DenseAdjacencyList
import .._Trees: TreeNode, Tree
import ..Algorithms: DiskANNIndex

# stdlib
import Mmap

# deps
import LightGraphs
import ProgressMeter
import StaticArrays: SVector
import UnPack: @unpack

export load, save, load_bin, save_bin
include("native.jl")

# Support for DiskANN generated binary files.
include("diskann.jl")
include("sptag.jl")

# Support for the standard "*.[b|i|c]vecs" formats.
include("vecs.jl")

end # module

