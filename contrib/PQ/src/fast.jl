#####
##### Permutation Intrinsic.
#####

function shuffle(a::SIMD.Vec{32,UInt16}, b::SIMD.Vec{32,UInt16}, idx::SIMD.Vec{32,UInt16})
    Base.@_inline_meta

    str = """
    declare <32 x i16> @llvm.x86.avx512.vpermi2var.hi.512(<32 x i16>, <32 x i16>, <32 x i16>) #2

    define <32 x i16> @entry(<32 x i16>, <32 x i16>, <32 x i16>) #0 {
    top:
        %val = tail call <32 x i16> @llvm.x86.avx512.vpermi2var.hi.512(<32 x i16> %0, <32 x i16> %1, <32 x i16> %2) #3
        ret <32 x i16> %val
    }

    attributes #0 = { alwaysinline }
    """

    x = Base.llvmcall(
        (str, "entry"),
        SIMD.LVec{32,UInt16},
        Tuple{SIMD.LVec{32,UInt16},SIMD.LVec{32,UInt16},SIMD.LVec{32,UInt16}},
        a.data,
        idx.data,
        b.data,
    )

    return SIMD.Vec(x)
end

# Dispatch Plumbing to take care of promotion.
function lookup(table::NTuple{8,SIMD.Vec{32,UInt16}}, queries::NTuple)
    return lookup(table, SIMD.Vec(queries))
end

function lookup(table::NTuple{8,SIMD.Vec{32,UInt16}}, queries::SIMD.Vec{32,UInt8})
    return lookup(table, convert(SIMD.Vec{32,UInt16}, queries))
end

function lookup(table::NTuple{8,SIMD.Vec{32,UInt16}}, queries::SIMD.Vec{32,UInt16})
    # Generate the masks for the blends.
    m1 = (queries & (UInt8(1) << 7)) > one(UInt8)
    m2 = (queries & (UInt8(1) << 6)) > one(UInt8)

    # Perform the mixing shuffles
    s1 = shuffle(table[1], table[2], queries)
    s2 = shuffle(table[3], table[4], queries)
    s3 = shuffle(table[5], table[6], queries)
    s4 = shuffle(table[7], table[8], queries)

    # Now reduce!
    t1 = SIMD.vifelse(m2, s2, s1)
    t2 = SIMD.vifelse(m2, s4, s3)
    return SIMD.vifelse(m1, t2, t1)
end

#####
##### Fast Table
#####

struct FastTable{T,N}
    # Compressed distances after the initial
    compressed::Vector{NTuple{8,SIMD.Vec{32,UInt16}}}
    table::DistanceTable{N,GraphANN.InnerProduct}
end

# TODO: This is will probably only really work for the InnerProduct metric at the moment.
function precompute!(table::FastTable, query::SVector)
    # Poor man's size check for now.
    @assert size(table.table.distances, 1) == 256

    # First, we perform the standard pre-computation step and calculate the metric between
    # the query and all of the centroids.
    #
    # TODO: We can compute the minimum value for each partition as we compute the initial
    # metric values. We should eventually do that.
    precompute!(table.table, query)

    # Now - we compute the minimum possible value that the quantized tables can take.
    distances = table.table.distances
    minval = sum(minimum, eachcol(distances))

    # This is the step size of our unsigned uncoding of the distances.
    Δ = -minval / typemax(UInt16)

    # Quantize the distances into "compressed"
    # We're going to switch the sign of the values and wrap returned results in "reversed"
    # to get the ordering correct.

end
