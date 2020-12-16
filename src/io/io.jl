module _IO

export DiskANNLoader, load_graph, save_graph
export load_vecs

# local deps
import .._Base: MetaGraph, stdallocator
import .._Graphs: UniDirectedGraph

# deps
import LightGraphs
import ProgressMeter
import UnPack: @unpack

# Support for DiskANN generated binary files.
include("diskann.jl")

# Support for the standard "*.[b|i|c]vecs" formats.
include("vecs.jl")

end # module
