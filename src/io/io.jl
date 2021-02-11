module _IO

export load_graph, save_graph
export load_vecs, save_vecs

# local deps
import .._Base: MetaGraph, stdallocator, medioid
import .._Graphs: UniDirectedGraph, FlatAdjacencyList
import .._Trees: TreeNode, Tree

# deps
import LightGraphs
import ProgressMeter
import StaticArrays: SVector
import UnPack: @unpack

# Support for DiskANN generated binary files.
include("diskann.jl")
include("sptag.jl")

# Support for the standard "*.[b|i|c]vecs" formats.
include("vecs.jl")

end # module

