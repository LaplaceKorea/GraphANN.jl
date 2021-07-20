# Wrapper for Neighbors that uses the LSB of the distance field to track if the
# neighbor has been expanded/visited yet.
abstract type AbstractMaskType end
struct DistanceLSB <: AbstractMaskType end
struct IDMSB <: AbstractMaskType end

struct MaskWrap{T <: AbstractMaskType,I,D}
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
_Base.getdistance(x::MaskWrap{DistanceLSB}) = clearlsb(getdistance(x.neighbor))

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

getlsb(x::T) where {T <: Integer} = x & one(T)
getlsb(x::Float32) = getlsb(reinterpret(UInt32, x))
getlsb(x::Float64) = getlsb(reinterpret(UInt64, x))

@inline getmsb(x::T) where {T <: Integer} = x & bitrotate(one(T), -1)

clearlsb(x::T) where {T <: Integer} = x & ~(one(T))
clearlsb(x::Float32) = reinterpret(Float32, clearlsb(reinterpret(UInt32, x)))
clearlsb(x::Float64) = reinterpret(Float64, clearlsb(reinterpret(UInt64, x)))
@inline clearmsb(x::T) where {T <: Integer} = x & ~bitrotate(one(T), -1)

setlsb(x::T) where {T <: Integer} = x | one(T)
setlsb(x::Float32) = reinterpret(Float32, setlsb(reinterpret(UInt32, x)))
setlsb(x::Float64) = reinterpret(Float64, setlsb(reinterpret(UInt64, x)))

@inline setmsb(x::T) where {T <: Integer} = x | bitrotate(one(T), -1)

@inline function Base.isless(x::D, y::MaskWrap{T,I,D}) where {T,I,D}
    # use "<" to generate smaller code for floats
    return x < getdistance(y)
end
@inline function Base.isless(x::MaskWrap{T,I,D}, y::MaskWrap{T,I,D}) where {T,I,D}
    # use "<" to generate smaller code for floats
    return isless(getdistance(x), y)
end

#####
##### BestBuffer
#####

mutable struct BestBuffer{T,I,D}
    entries::Vector{MaskWrap{T,I,D}}
    # How many entries are currently valid in the vector.
    currentlength::Int
    maxlength::Int
    # The index of the lowest-distance entry that has not yet been visited.
    bestunvisited::Int
end

function BestBuffer{T,I,D}(maxlen::Integer) where {T,I,D}
    entries = Vector{MaskWrap{T,I,D}}(undef, maxlen)
    return BestBuffer{T,I,D}(entries, 0, maxlen, 1)
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

Base.length(x::BestBuffer) = x.currentlength
Base.maximum(x::BestBuffer) = unwrap(x.entries[x.currentlength])

Base.insert!(x::BestBuffer{T}, v::Neighbor) where {T} = insert!(x, MaskWrap{T}(v))
function Base.insert!(x::BestBuffer, v::MaskWrap)
    @unpack entries, currentlength, bestunvisited, maxlength = x
    i = 1
    dv = getdistance(v)
    while i <= currentlength
        isless(dv, @inbounds(entries[i])) && break
        i += 1
    end
    i > maxlength && return nothing

    shift!(x, i)
    @inbounds entries[i] = v

    # Update best unvisited.
    if i < bestunvisited
        x.bestunvisited = i
    end
    return nothing
end

function shift!(x::BestBuffer, i)
    @unpack entries, currentlength, maxlength = x
    droplast = currentlength == maxlength
    if droplast
        maxind = (maxlength - 1)
    else
        maxind = currentlength
        x.currentlength = currentlength + 1
    end

    j = maxind
    while j >= i
        @inbounds(entries[j+1] = entries[j])
        j -= 1
    end
    return nothing
end

function getcandidate!(x::BestBuffer)
    @unpack entries, bestunvisited, currentlength = x
    candidate = entries[bestunvisited]
    entries[bestunvisited] = visited(candidate)

    # Find the next unvisited candidate.
    i = bestunvisited + 1
    while i <= currentlength
        !isvisited(@inbounds(entries[i])) && break
        i += 1
    end
    x.bestunvisited = i
    return getid(candidate)
end

done(x::BestBuffer) = (x.bestunvisited > x.currentlength)

