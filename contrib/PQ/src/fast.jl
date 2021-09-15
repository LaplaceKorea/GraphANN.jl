#####
##### Permutation Intrinsic.
#####

function shuffle(a::SIMD.Vec{32,Int16}, b::SIMD.Vec{32,Int16}, idx::SIMD.Vec{32,UInt16})
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
        SIMD.LVec{32,Int16},
        Tuple{SIMD.LVec{32,Int16},SIMD.LVec{32,UInt16},SIMD.LVec{32,Int16}},
        a.data,
        idx.data,
        b.data,
    )

    return SIMD.Vec(x)
end

# Dispatch Plumbing to take care of promotion.
# NOTE: These functions don't quite make it through Julia's inline heuristics, so
# we need to give it a little help.
@inline function lookup(table::NTuple{8,SIMD.Vec{32,Int16}}, queries::NTuple)
    return lookup(table, SIMD.Vec(queries))
end

@inline function lookup(table::NTuple{8,SIMD.Vec{32,Int16}}, queries::SIMD.Vec{32,UInt8})
    return lookup(table, convert(SIMD.Vec{32,UInt16}, queries))
end

function lookup(table::NTuple{8,SIMD.Vec{32,Int16}}, queries::SIMD.Vec{32,UInt16})
    Base.@_inline_meta

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

struct FastTable{N}
    # Compressed distances after the initial
    compressed::Vector{NTuple{8,SIMD.Vec{32,Int16}}}
    query_buffer::Vector{SIMD.Vec{32,UInt8}}
    table::DistanceTable{N,GraphANN.InnerProduct}
end

function FastTable(centroids::AbstractMatrix{SVector{N,Float32}}) where {N}
    numpartitions = size(centroids, 2)
    compressed = Vector{NTuple{8,SIMD.Vec{32,Int16}}}(undef, numpartitions)
    query_buffer = Vector{SIMD.Vec{32,UInt8}}(undef, numpartitions)
    table = DistanceTable(centroids, GraphANN.InnerProduct())
    return FastTable(compressed, query_buffer, table)
end

GraphANN.costtype(::MaybeThreadLocal{FastTable}, args...) = Int16
GraphANN.costtype(::MaybeThreadLocal{FastTable}, ::AbstractVector) = Int16
GraphANN.costtype(::MaybeThreadLocal{FastTable}, ::Type{<:Any}) = Int16
GraphANN.costtype(::MaybeThreadLocal{FastTable}, ::Type{<:Any}, ::Type{<:Any}) = Int16
GraphANN.ordering(::FastTable) = Base.Reverse

@inline ptradd(ptr::Ptr{T}, i) where {T} = ptr + sizeof(T) * (i - 1)
function queue!(table::FastTable, query, row::Integer)
    # What row are we on.
    # TODO: Remove sizing checks.
    buffer = table.query_buffer
    base = pointer(buffer) + (row - 1) * sizeof(UInt8)
    @inbounds for i in Base.OneTo(length(query))
        ptr = ptradd(base, i)
        unsafe_store!(Ptr{UInt8}(ptr), query[i])
    end
end

function evaluate(table::FastTable)
    accum = zero(SIMD.Vec{32,Int16})
    compressed, query_buffer = table.compressed, table.query_buffer
    @inbounds for i in eachindex(compressed, query_buffer)
        partial = lookup(compressed[i], query_buffer[i])
        accum = SIMD.add_saturate(accum, partial)
    end
    return accum
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
    maxval = sum(maximum, eachcol(distances))

    # Quantize the distances into "compressed"
    # We're going to switch the sign of the values and wrap returned results in "reversed"
    # to get the ordering correct.
    compressed = table.compressed
    ptr = Ptr{Int16}(pointer(compressed))
    for partition in axes(distances, 2)
        @simd ivdep for centroid in axes(distances, 1)
            distance = distances[centroid, partition]
            clamped = max(typemin(Int16), distance * typemax(Int16) / maxval)
            unsafe_store!(ptr, unsafe_trunc(Int16, clamped), centroid)
        end
        ptr += sizeof(eltype(compressed))
    end
end

function GraphANN.prehook(table::FastTable, query::GraphANN.MaybePtr{SVector})
    precompute!(table, maybeload(query))
end

#####
##### Custom Search
#####

mutable struct PQDiskANNRunner{I,D,O} <: GraphANN.AbstractDiskANNRunner{I,D,O}
    search_list_size::Int64

    # Pre-allocated buffer for the search list
    buffer::GraphANN.Algorithms.BestBuffer{GraphANN.Algorithms.DistanceLSB,I,D,O}
    visited::GraphANN.FastSet{I}

    # Pre-allocated destination for partial distances
    distance_queue::Vector{D}
end

function PQDiskANNRunner{I,D}(
    search_list_size::Integer, ordering::O
) where {I,D,O<:Base.Ordering}
    buffer = GraphANN.Algorithms.BestBuffer{GraphANN.Algorithms.DistanceLSB,I,D}(
        search_list_size, ordering
    )
    visited = GraphANN.FastSet{I}()
    distance_queue = Vector{D}(undef, 32)
    runner = PQDiskANNRunner{I,D,O}(
        convert(Int, search_list_size), buffer, visited, distance_queue
    )
    return runner
end

function PQDiskANNRunner(index::GraphANN.DiskANNIndex, search_list_size)
    I = eltype(index.graph)
    D = GraphANN.costtype(index.metric, index.data)
    return PQDiskANNRunner{I,D}(search_list_size, GraphANN.ordering(index.metric))
end

# Customized Search Routine
function GraphANN._Base.search(
    algo::PQDiskANNRunner,
    index::GraphANN.DiskANNIndex,
    _::GraphANN.MaybePtr{AbstractVector{T}},
    start::GraphANN.Algorithms.StartNode = index.startnode;
    callbacks = GraphANN.Algorithms.DiskANNCallbacks(),
    metric::FastTable = GraphANN.getlocal(index.metric),
) where {T<:Number}
    empty!(algo)

    # Destructure argument
    @unpack graph, data = index
    @unpack distance_queue = algo
    queue!(metric, start.value, 1)
    initial_distance = evaluate(metric)[1]
    GraphANN.Algorithms.pushcandidate!(
        algo, GraphANN.Neighbor(algo, start.index, initial_distance)
    )

    @inbounds while !GraphANN.Algorithms.done(algo)
        p = GraphANN.getid(GraphANN.Algorithms.unsafe_peek(algo))
        neighbors = GraphANN.Algorithms.LightGraphs.outneighbors(graph, p)

        # Prefetch all new datapoints.
        # IMPORTANT: This is critical for performance!
        @inbounds for vertex in neighbors
            GraphANN.prefetch(data, vertex)
        end

        # Prune
        # Do this here to allow the prefetched vectors time to arrive in the cache.
        GraphANN.Algorithms.getcandidate!(algo)
        algmax = GraphANN.getdistance(maximum(algo))

        # Prefetch potential next neigbors.
        !GraphANN.Algorithms.done(algo) && GraphANN.unsafe_prefetch(
            graph, GraphANN.getid(GraphANN.Algorithms.unsafe_peek(algo))
        )

        # Distance computations
        # Queue up in groups of 32
        for range in GraphANN.batched(eachindex(neighbors), 32)
            vneighbors = view(neighbors, range)
            for i in eachindex(range)
                @inbounds queue!(metric, data[vneighbors[i]], i)
            end
            distances = evaluate(metric)

            @inbounds for i in eachindex(vneighbors)
                if Base.lt(algo, distances[i], algmax) || !GraphANN.Algorithms.isfull(algo)
                    GraphANN.Algorithms.maybe_pushcandidate!(
                        algo, GraphANN.Neighbor(algo, vneighbors[i], distances[i])
                    )
                    algmax = GraphANN.Algorithms.getdistance(maximum(algo))
                end
            end
        end

        callbacks.postdistance(algo, p, neighbors)
    end

    return nothing
end

#####
##### Custom Reranking
#####

# function (reranker::Reranker)(
#     runner::PQDiskANNRunner, num_neighbors, query
# )
#     @unpack dataset, metric = reranker
#     @unpack buffer = runner
#     @unpack entries = buffer
#
#     # Find the maximum distance in entries for quantization purposes.
#     mindistance, maxdistance = extrema(GraphANN.getdistance, entries)
#     delta = maxdistance - mindistance
#
#     function scale(x)
#         v = (x - mindistance) * typemax(Int16) / delta
#         @show (mindistance, maxdistance, delta, x, v)
#         return unsafe_trunc(Int16, max(v, typemin(Int16)))
#     end
#
#     # Populate the full distance for each candidate.
#     for i in eachindex(entries)
#         id = GraphANN.getid(entries[i])
#         distance = GraphANN.evaluate(metric, query, pointer(dataset, id))
#         @inbounds GraphANN.Algorithms.unsafe_replace!(buffer, i, id, scale(distance))
#     end
#
#     # Sort based on distance.
#     return partialsort!(entries, Base.OneTo(num_neighbors), GraphANN.ordering(metric))
# end

