module _IO

export DiskANNLoader, load_graph, save_graph
export load_vecs, save_vecs

# local deps
import .._Base: MetaGraph, stdallocator, medioid
import .._Graphs: UniDirectedGraph
import .._Trees: BKTNode, BKTree

# deps
import LightGraphs
import ProgressMeter
import UnPack: @unpack

# Support for DiskANN generated binary files.
include("diskann.jl")
include("sptag.jl")

# Support for the standard "*.[b|i|c]vecs" formats.
include("vecs.jl")

end # module

