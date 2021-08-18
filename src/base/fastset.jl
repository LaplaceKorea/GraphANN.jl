sentinel(::Type{T}) where {T <: Integer} = typemin(T)
sentinel(::T) where {T} = sentinel(T)

issentinel(x::T) where {T} = (x == sentinel(x))

const global maxallowedprobe = 16
const global maxprobeshift   = 6

mutable struct FastSet{K} <: AbstractSet{K}
    keys::Array{K,1}
    count::Int
    age::UInt
    idxfloor::Int
    maxprobe::Int

    function FastSet{K}() where {K}
        new(fill(sentinel(K), 16), 0, 0, 1, 0)
    end

    function FastSet{K}(d::FastSet{K}) where {K}
        new(copy(d.keys), d.count, d.age, d.idxfloor, d.maxprobe)
    end

    function FastSet{K}(keys, count, age, idxfloor, maxprobe) where {K}
        new(keys, count, age, idxfloor, maxprobe)
    end
end

Base.length(h::FastSet) = h.count

#####
##### Shadow a bunch of Base methods
#####

_tablesz(x::Integer) = x < 16 ? 16 : one(x)<<((sizeof(x)<<3)-leading_zeros(x-1))
hashindex(key, sz) = (((hash(key)::UInt % Int) & (sz-1)) + 1)::Int

Base.@propagate_inbounds isslotfilled(h::FastSet, i::Int) = !issentinel(h.keys[i])
Base.@propagate_inbounds isslotempty(h::FastSet, i::Int) = issentinel(h.keys[i])

function rehash!(h::FastSet{K}, newsz = length(h.keys)) where {K}
    oldk = h.keys
    sz = length(oldk)
    newsz = _tablesz(newsz)
    h.age += 1
    h.idxfloor = 1
    if h.count == 0
        resize!(h.keys, newsz)
        h.keys .= sentinel(K)
        return h
    end

    keys = fill(sentinel(K), newsz)
    age0 = h.age
    count = 0
    maxprobe = 0

    for i in Base.OneTo(sz)
        @inbounds if !issentinel(oldk[i])
            k = oldk[i]
            index0 = index = hashindex(k, newsz)
            while !issentinel(keys[index])
                index = (index & (newsz-1)) + 1
            end
            probe = (index - index0) & (newsz-1)
            probe > maxprobe && (maxprobe = probe)
            keys[index] = k
            count += 1

            if h.age != age0
                # if `h` is changed by a finalizer, retry
                return rehash!(h, newsz)
            end
        end
    end

    h.keys = keys
    h.count = count
    h.maxprobe = maxprobe
    @assert h.age == age0
    return h
end

function Base.empty!(h::FastSet{K}) where {K}
    h.keys .= sentinel(K)
    h.count = 0
    h.age += 1
    h.idxfloor = 1
    return h
end

# get the index where a key is stored, or -1 if not present
function ht_keyindex(h::FastSet{K}, key) where {K}
    sz = length(h.keys)
    iter = 0
    maxprobe = h.maxprobe
    index = hashindex(key, sz)
    keys = h.keys

    @inbounds while true
        if isslotempty(h,index)
            break
        end
        if key === keys[index] || isequal(key, keys[index])
            return index
        end

        index = (index & (sz-1)) + 1
        iter += 1
        iter > maxprobe && break
    end
    return -1
end

# get the index where a key is stored, or -pos if not present
# and the key would be inserted at pos
# This version is for use by setindex! and get!
function ht_keyindex2!(h::FastSet{K}, key) where {K}
    age0 = h.age
    sz = length(h.keys)
    iter = 0
    maxprobe = h.maxprobe
    index = hashindex(key, sz)
    avail = 0
    keys = h.keys

    @inbounds while true
        if isslotempty(h,index)
            if avail < 0
                return avail
            end
            return -index
        end

        if key === keys[index] || isequal(key, keys[index])
            return index
        end

        index = (index & (sz-1)) + 1
        iter += 1
        iter > maxprobe && break
    end

    avail < 0 && return avail

    maxallowed = max(maxallowedprobe, sz>>maxprobeshift)
    # Check if key is not present, may need to keep searching to find slot
    @inbounds while iter < maxallowed
        if !isslotfilled(h,index)
            h.maxprobe = iter
            return -index
        end
        index = (index & (sz-1)) + 1
        iter += 1
    end

    rehash!(h, h.count > 64000 ? sz*2 : sz*4)
    return ht_keyindex2!(h, key)
end

Base.@propagate_inbounds function _setindex!(h::FastSet{K}, key, index) where {K}
    h.keys[index] = key
    h.count += 1
    h.age += 1
    if index < h.idxfloor
        h.idxfloor = index
    end

    sz = length(h.keys)
    # Rehash now if necessary
    if h.count*3 > sz*2
        # > 2/3 full
        rehash!(h, h.count > 64000 ? h.count*2 : h.count*4)
    end
end

function Base.push!(h::FastSet{K}, key0) where {K}
    key = convert(K, key0)
    if !isequal(key, key0)
        throw(ArgumentError("$(limitrepr(key0)) is not a valid key for type $K"))
    end
    setindex!(h, key)
end

function Base.setindex!(h::FastSet{K}, key::K) where {K}
    index = ht_keyindex2!(h, key)

    if index > 0
        h.age += 1
        @inbounds h.keys[index] = key
    else
        @inbounds _setindex!(h, key, -index)
    end
    return h
end

Base.haskey(h::FastSet, key) = (ht_keyindex(h, key) >= 0)
Base.in(key, h::FastSet) = (ht_keyindex(h, key) >= 0)

Base.@propagate_inbounds function Base.iterate(h::FastSet, i = 1)
    len = length(h.keys)
    while true
        i > len && return nothing
        if isslotfilled(h, i)
            return (h.keys[i], i + 1)
        end
        i += 1
    end
end
