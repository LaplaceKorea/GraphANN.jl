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

first_available(::T, word::UInt64) where {T <: Bucket} = first_available(T, word)
function first_available(::Type{Bucket{F,B}}, word::UInt64) where {F,B}
    lz = leading_zeros(word) - padding_bits(Bucket{F,B})
    return one(UInt64) + fingerprints_per_word(Bucket{F,B}) - div(lz, F)
end

function space_available(::Type{Bucket{F,B}}, word::UInt64) where {F,B}
    return first_available(Bucket{F,B}, word) <= fingerprints_per_word(Bucket{F,B})
end

adjust_shift(::Type{Bucket{F,B}}, shift) where {F,B} = (F * (shift - 1))
function _insert(::Type{T}, word::UInt64, fingerprint::UInt64, shift) where {T <: Bucket}
    return word | (fingerprint << adjust_shift(T, shift))
end

function _insert_and_kick(
    ::Type{T}, word::UInt64, fingerprint::UInt64, shift_amount
) where {T <: Bucket}
    mask = fingerprintmask(T)
    old_fingerprint = (word >> shift_amount) & mask
    return (old_fingerprint, _insert(T, word, fingerprint, shift_amount))
end

function imprint(x, ::Bucket{F,B}) where {F,B}
    h = hash(x, FINGERPRINT_SALT)
    fingerprint = h & fingerprintmask(Bucket{F,B})
    # fingerprint cannot be zero
    while fingerprint == typemin(UInt64) # Must not be zero
        h = h >>> F + 1
        fingerprint = h & fingerprintmask(Bucket{F,B})
    end
    return UInt64(fingerprint)
end

### High Level API
function Base.in(fingerprint, bucket::Bucket{F,B}) where {F,B}
    @unpack ptr = bucket
    for i in Base.OneTo(B)
        word = unsafe_load(ptr, i)
        inword(Bucket{F,B}, fingerprint, word) && return true
    end
    return false
end

function trypush!(bucket::T, fingerprint) where {F, B, T <: Bucket{F,B}}
    @unpack ptr = bucket
    for i in Base.OneTo(B)
        word = unsafe_load(ptr, i)
        # If this fingerprint already exists, do nothing.
        # We can do this because we don't support deletion.
        inword(T, fingerprint, word) && return true

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

#####
##### CuckooFilter
#####

struct CuckooFilter{F,B}
    data::Vector{UInt64}
    mask::UInt64
end

buckettype(::CuckooFilter{F,B}) where {F,B} = Bucket{F,B}

function primaryindex(filter::CuckooFilter, x)
    return (hash(x) & filter.mask) + 1
end

function otherindex(filter::CuckooFilter, fingerprint::UInt64, i) where {T <: Bucket}
    return xor(i - 1, hash(fingerprint)) & filter.mask + 1
end

function getbucket(filter::CuckooFilter{F,B}, i) where {F,B}
    return Bucket{F,B}(pointer(filter.data, B * (i - 1) + 1))
end

function Base.in(filter::CuckooFilter, x)
    fingerprint = imprint(buckettype(filter), x)

    # Check the primary location
    primary = primaryindex(filter, x)
    in(fingerprint, getbucket(filter, primary)) && return true

    # Check the other location.
    other = otherindex(filter, fingerprint, first_index)
    in(fingerprint, getbucket(filter, other)) && return true

    # TODO: Stash for "emergencies"?
    return false
end


