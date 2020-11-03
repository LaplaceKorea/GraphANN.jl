# Extensions to DataStructures.jl

#####
##### Robin Set
#####

# Construct a RobinSet by using a thin wrapper around a RobinDict
#
# Implementation roughly based on Base's implementation of Set.
struct RobinSet{T} <: AbstractSet{T}
    dict::DataStructures.RobinDict{T,Nothing}

    # Inner Constructors
    RobinSet{T}() where {T} = new(Dict{T,Nothing}())
end

Base.push!(set::RobinSet, k) = (set.dict[k] = nothing)
Base.pop!(set::RobinSet) = first(pop!(set.dict))
Base.in(k, set::RobinSet) = haskey(set.dict, k)
Base.delete!(set::RobinSet, k) = delete!(set.dict, k)
Base.empty!(set::RobinSet) = empty!(set.dict)
