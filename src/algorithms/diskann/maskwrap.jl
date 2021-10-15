#####
##### Bit Manipulations
#####

# Get bit set info
getlsb(x::T) where {T<:Integer} = x & one(T)
getlsb(x::Float32) = getlsb(reinterpret(UInt32, x))
getlsb(x::Float64) = getlsb(reinterpret(UInt64, x))
getmsb(x::T) where {T<:Integer} = x & bitrotate(one(T), -1)

# Clear bits
clearlsb(x::T) where {T<:Integer} = x & ~(one(T))
clearlsb(x::Float32) = reinterpret(Float32, clearlsb(reinterpret(UInt32, x)))
clearlsb(x::Float64) = reinterpret(Float64, clearlsb(reinterpret(UInt64, x)))
clearmsb(x::T) where {T<:Integer} = x & ~bitrotate(one(T), -1)

# Set bits
setlsb(x::T) where {T<:Integer} = x | one(T)
setlsb(x::Float32) = reinterpret(Float32, setlsb(reinterpret(UInt32, x)))
setlsb(x::Float64) = reinterpret(Float64, setlsb(reinterpret(UInt64, x)))
setmsb(x::T) where {T<:Integer} = x | bitrotate(one(T), -1)

#####
##### Mask Wrap
#####

# Wrapper for Neighbors that uses the LSB of the distance field to track if the
# neighbor has been expanded/visited yet.
abstract type AbstractMaskType end
struct DistanceLSB <: AbstractMaskType end
struct IDMSB <: AbstractMaskType end

struct MaskWrap{T<:AbstractMaskType,I,D}
    neighbor::Neighbor{I,D}
    MaskWrap{T,I,D}(neighbor::Neighbor{I,D}) where {T,I,D} = new{T,I,D}(neighbor)
end

function MaskWrap{DistanceLSB}(x::Neighbor{I,D}) where {I,D}
    return MaskWrap{DistanceLSB,I,D}(Neighbor{I,D}(x.id, clearlsb(x.distance)))
end

function MaskWrap{IDMSB}(x::Neighbor{I,D}) where {I,D}
    return MaskWrap{IDMSB,I,D}(Neighbor{I,D}(clearmsb(x.id), x.distance))
end

unwrap(x::MaskWrap) = x.neighbor
_Base.getid(x::MaskWrap) = getid(x.neighbor)
@inline _Base.getid(x::MaskWrap{IDMSB}) = clearmsb(getid(x.neighbor))
_Base.getdistance(x::MaskWrap) = getdistance(x.neighbor)

# NOTE: Don't need this method in order to be correct.
# Thus, we don't need to mask whenever we unwrap distance fields.
#_Base.getdistance(x::MaskWrap{DistanceLSB}) = clearlsb(getdistance(x.neighbor))

isvisited(x::MaskWrap{DistanceLSB}) = isone(getlsb(getdistance(x.neighbor)))
@inline isvisited(x::MaskWrap{IDMSB}) = isone(getmsb(getid(x.neighbor)))

function visited(x::MaskWrap{DistanceLSB,I,D}) where {I,D}
    @unpack id, distance = x.neighbor
    return MaskWrap{DistanceLSB,I,D}(Neighbor{I,D}(id, setlsb(distance)))
end

function visited(x::MaskWrap{IDMSB,I,D}) where {I,D}
    @unpack id, distance = x.neighbor
    return MaskWrap{IDMSB,I,D}(Neighbor{I,D}(setmsb(id), distance))
end

Base.isless(x::D, y::MaskWrap{T,I,D}) where {T,I,D} = x < getdistance(y)
Base.isless(x::MaskWrap{T,I,D}, y::D) where {T,I,D} = getdistance(x) < y
function Base.isless(x::MaskWrap{T,I,D}, y::MaskWrap{T,I,D}) where {T,I,D}
    return isless(getdistance(x), y)
end

#####
##### BestBuffer
#####

mutable struct BestBuffer{T,I,D,O <: Base.Ordering}
    entries::Vector{MaskWrap{T,I,D}}
    # How many entries are currently valid in the vector.
    currentlength::Int
    maxlength::Int
    # The index of the lowest-distance entry that has not yet been visited.
    bestunvisited::Int
    ordering::O
end
@inline Base.lt(o::BestBuffer, x, y) = Base.lt(o.ordering, x, y)

function BestBuffer{T,I,D}(maxlen::Integer, ordering::O) where {T,I,D,O}
    entries = Vector{MaskWrap{T,I,D}}(undef, maxlen)
    return BestBuffer{T,I,D,O}(entries, 0, maxlen, 1, ordering)
end

function Base.empty!(buffer::BestBuffer)
    buffer.currentlength = 0
    buffer.bestunvisited = 1
    return nothing
end

function Base.resize!(buffer::BestBuffer, val)
    resize!(buffer.entries, val)
    buffer.maxlength = val
    return nothing
end

Base.length(buffer::BestBuffer) = buffer.currentlength
Base.maximum(buffer::BestBuffer) = unwrap(buffer.entries[buffer.currentlength])

Base.insert!(buffer::BestBuffer{T}, v::Neighbor) where {T} = insert!(buffer, MaskWrap{T}(v))
function Base.insert!(buffer::BestBuffer, v::MaskWrap)
    @unpack entries, currentlength, maxlength, bestunvisited = buffer
    i = 1
    dv = getdistance(v)
    while i <= currentlength
        Base.lt(buffer, dv, @inbounds entries[i]) && break
        i += 1
    end
    i > maxlength && return false

    shift!(buffer, i)
    @inbounds entries[i] = v

    # Update best unvisited.
    if i < bestunvisited
        buffer.bestunvisited = i
        return true
    end
    return false
end

function shift!(buffer::BestBuffer, i)
    @unpack entries, currentlength, maxlength = buffer
    droplast = currentlength == maxlength
    if droplast
        maxind = (maxlength - 1)
    else
        maxind = currentlength
        buffer.currentlength = currentlength + 1
    end

    j = maxind
    @inbounds while j >= i
        entries[j + 1] = entries[j]
        j -= 1
    end
    return nothing
end

function getcandidate!(buffer::BestBuffer)
    @unpack entries, currentlength, bestunvisited = buffer
    candidate = entries[bestunvisited]
    entries[bestunvisited] = visited(candidate)

    # Find the next unvisited candidate.
    i = bestunvisited + 1
    while i <= currentlength
        !isvisited(@inbounds entries[i]) && break
        i += 1
    end
    buffer.bestunvisited = i
    return getid(candidate)
end

done(buffer::BestBuffer) = (buffer.bestunvisited > buffer.currentlength)

function unsafe_replace!(buffer::BestBuffer{T,I,D}, i, id, distance) where {T,I,D}
    wrap = MaskWrap{T,I,D}(Neighbor{I,D}(id, convert(D, distance)))
    buffer.entries[i] = wrap
    return nothing
end

