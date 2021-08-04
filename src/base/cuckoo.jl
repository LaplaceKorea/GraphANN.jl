# Wrapper for the start of a bucket.
# The bucket consists of "F"-bit fingerprints taking "B" UInt64 words.
const SALT = UInt64(0x657a_8b92_0eef_9adc)
struct Bucket{F,B}
    ptr::Ptr{UInt64}
end

Bucket{F,B}() where {F,B} = Bucket{F,B}(Ptr{UInt64}())

@inline bitsize(::Type{T}) where {T} = 8 * sizeof(T)
padding_bits(::Type{Bucket{F,B}}) where {F,B} = rem(bitsize(UInt64), F)
fingerprints_per_word(::Type{Bucket{F,B}}) where {F,B} = div(bitsize(UInt64), F)

Base.length(::Type{Bucket{F,B}}) where {F,B} = B * fingerprints_per_word(Bucket{F,B})
Base.length(::T) where {T <: Bucket} = length(T)

_size(::Type{Bucket{F,B}}) where {F,B} = (fingerprints_per_word(Bucket{F,B}), B)
_size(::T) where {T <: Bucket} = _size(T)

# NOTE: This should be fully constant propagated.
# If it isn't in future versions, it should be implemented as `@generated`.
searchmask(x::T) where {T<:Bucket} = searchmask(T)
function searchmask(::Type{Bucket{F,B}}) where {F,B}
    # How many fingerprints are packed into a Word.
    fingerprints_per_word = div(bitsize(UInt64), F)
    s = zero(UInt64)
    for i in Base.OneTo(fingerprints_per_word)
        s += one(UInt64) << (F * (i - 1))
    end
    return s
end

fingerprintmask(::Type{Bucket{F,B}}) where {F,B} = (one(UInt64) << F) - one(UInt64)
fingerprintmask(::T) where {T <: Bucket} = fingerprintmask(T)

function inword(::Type{T}, fingerprint::UInt64, word::UInt64) where {T<:Bucket}
    Base.@_inline_meta
    return !iszero(_inword(T, fingerprint, word))
end

function _inword(::Type{T}, fingerprint::UInt64, word::UInt64) where {F,T<:Bucket{F}}
    Base.@_inline_meta
    smask = searchmask(T)
    q = xor(word, xor(fingerprint, fingerprintmask(T)) * smask)

    # Need to carefully handle the case where we match the top most slot and generate
    # a carry out.
    #
    # The general idea is to do a checked add with overflow, then rotate everything
    # right one bit to accomodate that uppermost carry out.
    #
    # Since we don't care about the low bits anyways, this works just fine.
    (s, f) = Base.add_with_overflow(q, smask)
    s = (s >> 1) | (UInt(f) << 63)
    mulmask = (((one(UInt64) << F) * smask) >> 1) | one(UInt64) << 63
    r = xor(s, q >> 1, smask >> 1) & mulmask
    return r
end

first_available(::T, word::UInt64) where {T<:Bucket} = first_available(T, word)
function first_available(::Type{Bucket{F,B}}, word::UInt64) where {F,B}
    lz = leading_zeros(word) - padding_bits(Bucket{F,B})
    return one(UInt64) + fingerprints_per_word(Bucket{F,B}) - div(lz, F)
end

function space_available(::Type{Bucket{F,B}}, word::UInt64) where {F,B}
    return first_available(Bucket{F,B}, word) <= fingerprints_per_word(Bucket{F,B})
end

@inline adjust(::Type{Bucket{F,B}}, shift) where {F,B} = (F * (shift - 1))
function _insert(::Type{T}, word::UInt64, fingerprint::UInt64, shift) where {T<:Bucket}
    return word | (fingerprint << adjust(T, shift))
end

function _insert_and_kick(
    ::Type{T}, word::UInt64, fingerprint::UInt64, shift
) where {T<:Bucket}
    mask = fingerprintmask(T)
    old_fingerprint = (word >> adjust(T, shift)) & mask
    word &= ~(mask << adjust(T, shift))
    return (old_fingerprint, _insert(T, word, fingerprint, shift))
end

### High Level API
function Base.in(fingerprint, bucket::T) where {F,B,T<:Bucket{F,B}}
    Base.@_inline_meta
    @unpack ptr = bucket
    for i in Base.OneTo(B)
        word = unsafe_load(ptr, i)
        inword(T, fingerprint, word) && return true
    end
    return false
end

function trypush!(bucket::T, fingerprint) where {F,B,T<:Bucket{F,B}}
    @unpack ptr = bucket
    for i in Base.OneTo(B)
        word = unsafe_load(ptr, i)
        # If this fingerprint already exists, do nothing.
        # We can do this because we don't support deletion.
        if inword(T, fingerprint, word)
            return true
        end

        # See if there is space available in this word.
        # If so, store the fingerprint in the first available slot.
        # Otherwise, we need to move onto the next word.
        shift_amount = first_available(T, word)
        if shift_amount <= fingerprints_per_word(T)
            word = _insert(T, word, fingerprint, shift_amount)
            unsafe_store!(ptr, word, i)
            return true
        end
    end
    return false
end

# Unsafe because we won't bounds check the `big` and `little` indices.
function unsafe_replace!(bucket::T, fingerprint, big, little) where {T}
    @unpack ptr = bucket
    word = unsafe_load(ptr, big)
    old, newword = _insert_and_kick(T, word, fingerprint, little)
    unsafe_store!(ptr, newword, big)
    return old
end

function imprint(::Type{Bucket{F,B}}, x) where {F,B}
    h = hash(x, SALT)
    fingerprint = h & fingerprintmask(Bucket{F,B})
    # fingerprint cannot be zero
    while iszero(fingerprint)
        h = (h >>> F) + 1
        fingerprint = h & fingerprintmask(Bucket{F,B})
    end
    return UInt64(fingerprint)
end

#####
##### CuckooFilter
#####

# Make a global constant for now - revisit later.
const MAX_KICKS = 512

struct CuckooFilter{F,B}
    data::Vector{UInt64}
    mask::UInt64
end

function CuckooFilter{F,B}(len::Integer) where {F,B}
    @assert ispow2(len)
    data = zeros(UInt64, B * len)
    mask = UInt(len - 1)
    return CuckooFilter{F,B}(data, mask)
end

buckettype(::CuckooFilter{F,B}) where {F,B} = Bucket{F,B}

function primaryindex(filter::CuckooFilter, x)
    return (hash(x) & filter.mask) + 1
end

function otherindex(filter::CuckooFilter, fingerprint::UInt64, i)
    return xor(i - 1, hash(fingerprint)) & filter.mask + 1
end

function getbucket(filter::CuckooFilter{F,B}, i) where {F,B}
    ptr = Base.unsafe_convert(Ptr{UInt64}, filter.data) + sizeof(UInt64) * B * (i-1)
    return Bucket{F,B}(ptr)
end

Base.in(x, filter::CuckooFilter) = _in(imprint(buckettype(filter), x), filter)
function _in(fingerprint::UInt64, filter::CuckooFilter)
    # Check the primary location
    primary = primaryindex(filter, fingerprint)
    in(fingerprint, getbucket(filter, primary)) && return true

    # Check the other location.
    other = otherindex(filter, fingerprint, primary)
    in(fingerprint, getbucket(filter, other)) && return true
    return false
end

randbig(::CuckooFilter{F,B}) where {F,B} = rand(ntuple(identity, Val(B)))
function randlittle(::CuckooFilter{F,B}) where {F,B}
    return rand(ntuple(identity, Val(fingerprints_per_word(Bucket{F,B}))))
end
randindex(filter::CuckooFilter) = (randbig(filter), randlittle(filter))

Base.push!(filter::CuckooFilter, x) = _push!(filter, imprint(buckettype(filter), x))
function _push!(filter::CuckooFilter{F,B}, fingerprint::UInt64) where {F,B}
    primary = primaryindex(filter, fingerprint)
    trypush!(getbucket(filter, primary), fingerprint) && return true

    other = otherindex(filter, fingerprint, primary)
    trypush!(getbucket(filter, other), fingerprint) && return true

    # Both failed, time to start getting serious
    index = primary
    bucket = getbucket(filter, index)
    tripcount = 0
    while true
        big, little = randindex(filter)
        fingerprint = unsafe_replace!(bucket, fingerprint, big, little)
        index = otherindex(filter, fingerprint, index)

        # Prepare for next iteration.
        bucket = getbucket(filter, index)
        trypush!(bucket, fingerprint) && break
        tripcount += 1
        if tripcount == MAX_KICKS
            error("Insertion Failed!")
        end
    end
    return true
end

