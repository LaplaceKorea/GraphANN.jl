module Regression

# You know - the thing we're actually testing.
using GraphANN: GraphANN

# stdlib
using Serialization
using Statistics

# deps
using DataFrames
using DataStructures

using PrettyTables: PrettyTables
import UnPack: @unpack, @pack!

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const SCRATCH = joinpath(PKGDIR, "data")

makescratch() = ispath(SCRATCH) || mkpath(SCRATCH)

include("utils.jl")
include("types.jl")
include("index.jl")
include("query.jl")
include("clustering.jl")
include("test.jl")
include("dump.jl")

end # module
