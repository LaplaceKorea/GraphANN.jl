module _Trees

# local deps
using .._Base

# deps
using DataStructures: DataStructures
import UnPack: @unpack, @pack!

abstract type AbstractTree end

export AbstractTree
export TreeNode, isleaf, childindices
export Tree
include("tree.jl")

end
