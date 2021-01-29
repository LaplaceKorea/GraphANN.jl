module _Trees

# local deps
using .._Base

# deps
import DataStructures
import UnPack: @unpack

export TreeNode, isleaf, childindices
export Tree
include("tree.jl")

end
