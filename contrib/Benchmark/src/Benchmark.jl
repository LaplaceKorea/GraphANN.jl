module Benchmark

import GraphANN

# stdlib
using Serialization
using Statistics

# deps
using DataFrames
using DataStructures
using ProgressMeter

import PrettyTables
import UnPack: @unpack, @pack!

# paths
const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const SCRATCH = joinpath(PKGDIR, "data")

# init
makescratch() = ispath(SCRATCH) || mkpath(SCRATCH)
function __init__()
    makescratch()
end
include("types.jl")

# routines
include("sptag.jl")

end # module
