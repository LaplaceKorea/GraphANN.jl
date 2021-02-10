module GraphANN

# We use StaticArrays pretty heavily in the code, so import the `SVector` symbol
# so users can access this type via `GraphANN.SVector`.
import StaticArrays: SVector

# Bootstrap
include("base/base.jl"); using ._Base
include("graphs/graphs.jl"); using ._Graphs
include("trees/trees.jl"); using ._Trees
include("points/points.jl"); using ._Points
include("prefetch/prefetch.jl"); using ._Prefetcher
include("quantization/quantization.jl"); using ._Quantization
include("io/io.jl"); using ._IO

# Core implementation
include("algorithms/algorithms.jl"); using .Algorithms

end # module
