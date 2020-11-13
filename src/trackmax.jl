#####
##### Track Max
#####

# For understanding how large to create auxiliary data structures during the initial
# allocation, we want to track the largest size of each seen.
#
# This leads to the conundrum:
#
# 1. How do we support tracking this size information for some runs, but not others.
# 2. How do we maintain this feature instead of making it a one-off or something?
#
# The answer - we use the wonderful "TrackMax" structure below!

struct TrackMax{T}
    val::T
    maxes::Dict{Symbol,Int}
end

TrackMax(val::T) where {T} = TrackMax{T}(val, Dict{Symbol,Int}())

unwrap(x) = x
unwrap(x::TrackMax) = x.val

# When used with ThreadLocal storage, we want to return the unwrapped types themselves.
function tlshook(x::TrackMax{<:NamedTuple{names}}) where {names}
    track!(x)
    x = unwrap(x)
    return NamedTuple{names}(unwrap.(Tuple(x)))
end

# Default behavior - do nothing.
# Specialize for data types of interest.
track!(x) = nothing

# Recurse over NamedTuples
track!(x::TrackMax{<:Tuple}) = track!.(unwrap(x))
track!(x::TrackMax{<:NamedTuple}) = track!.(Tuple(unwrap(x)))

# Provide a summary
function summarize(x::TrackMax{<:NamedTuple{names}}) where {names}
    summary = Dict{Symbol,Any}()
    for name in names
        summary[name] = summarize(x.val[name])
    end
    return summary
end

summarize(x::TrackMax) = x.maxes
summarize(x) = nothing

# Purpose of macro
#
# @trackmax items expr
#
# becomes
#
# v = expr
# if v > maxes[items]
#   maxes[items] = v
# end
#
macro trackmax(name, k, expr)
    k = QuoteNode(k)
    name = esc(name)
    return quote
        v = $(esc(expr))
        if v > get!($name, $k, 0)
            $name[$k] = v
        end
    end
end

# Specializations
function track!(x::TrackMax{<:Pruner})
    @unpack val, maxes = x
    @trackmax maxes items length(val.items)
end

function track!(x::TrackMax{<:NextListBuffer})
    @unpack val, maxes = x
    @unpack buffers, nextlists = val

    @trackmax maxes num_buffers length(buffers)
    @trackmax maxes buffer_length begin
        length(nextlists) == 0 ? 0 : maximum(length, values(nextlists))
    end
    @trackmax maxes nextlist length(nextlists)
end
