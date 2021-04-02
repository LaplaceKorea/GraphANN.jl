# Put all the Query.jl related stuff in 1 file so we can easily include it
# or not if we want Query.jl stuff
using Query

macro sub(df, sym)
    sym = QuoteNode(sym)
    return :($sym => $(esc(df))[:, $sym])
end

#####
##### Indexing Performance
#####

format_index(record::Record) = format_index(record.df)
function format_index(df::DataFrame)
    unique_maxlines = unique(df[:, :maxlines])
    dest = DataFrame()

    for maxlines in unique_maxlines
        _format_index!(dest, df, maxlines)
    end
    return dest
end

function _format_index!(dest::DataFrame, src::DataFrame, maxlines)
    # Filter so we only get the matching items.
    subdf = @from i in src begin
        @where i.maxlines == maxlines
        @select i
        @collect DataFrame
    end

    d = OrderedDict(
        @sub(subdf, time),
        :alloc_gb => subdf[:, :bytes] ./ 1E9,
        @sub(subdf, graph_allocator),
        @sub(subdf, data_allocator),
        @sub(subdf, target_degree),
        @sub(subdf, window_size),
        @sub(subdf, num_threads),
        @sub(subdf, prune_to_degree),
        @sub(subdf, prune_threshold_degree),
        @sub(subdf, alpha),
        @sub(subdf, batchsize)
    )

    temp = DataFrame(d)
    for row in eachrow(temp)
        push!(dest, row; cols = :union)
    end
    return nothing
end

#####
##### Query Performance
#####

format_query(record::Record) = format_query(record.df)
function format_query(df::DataFrame)
    # Loop through num neighbors and recall.
    num_neighbors = unique(df[:, :num_neighbors])
    target_recalls = unique(df[:, :target_recall])

    dest = DataFrame()
    for nn in num_neighbors, target_recall in target_recalls
        _format_query!(dest, df, nn, target_recall)
    end
    return dest
end

function _format_query!(dest::DataFrame, src::DataFrame, nn::Integer, target_recall::Number)
    # Filter so we only get the matching items.
    subdf = @from i in src begin
        @where i.num_neighbors == nn && i.target_recall == target_recall
        @select i
        @collect DataFrame
    end

    d = OrderedDict(
        @sub(subdf, num_neighbors),
        @sub(subdf, target_recall),
        @sub(subdf, maxlines),
        @sub(subdf, recall),
        @sub(subdf, windowsize),
        @sub(subdf, qps),
        @sub(subdf, mean_latency),
        :latency_9 => getindex.(subdf[:, :nines_latency], 1),
        :latency_99 => getindex.(subdf[:, :nines_latency], 2),
        :latency_999 => getindex.(subdf[:, :nines_latency], 3),
        :latency_9999 => getindex.(subdf[:, :nines_latency], 4),
        @sub(subdf, max_latency),
        @sub(subdf, data_allocator),
        @sub(subdf, graph_allocator),
        @sub(subdf, prefetching),
        @sub(subdf, threading),
        @sub(subdf, num_threads),
    )

    temp = DataFrame(d)
    for row in eachrow(temp)
        push!(dest, row; cols = :union)
    end
    return nothing
end

#####
##### Clustering Results
#####

format_cluster(record::Record) = format_cluster(record.df)
function format_cluster(df::DataFrame)
    prefix = unique(df[:, :saveprefix])
    dest = DataFrame()

    for p in prefix
        _format_cluster!(dest, df, p)
    end
    return dest
end

function _format_cluster!(dest::DataFrame, src::DataFrame, prefix::AbstractString)
    subdf = @from i in src begin
        @where i.saveprefix == prefix
        @select i
        @collect DataFrame
    end

    d = OrderedDict(
        :model => prefix,
        @sub(subdf, savename),
        :num_partitions => div.(128, subdf.partition_size),
        @sub(subdf, partition_size),
        @sub(subdf, num_centroids),
        @sub(subdf, recall_1at1),
        @sub(subdf, recall_1at5),
        @sub(subdf, recall_1at10),
        @sub(subdf, recall_1at100),
        @sub(subdf, recall_5at5),
        @sub(subdf, recall_5at10),
        @sub(subdf, recall_5at20),
        @sub(subdf, recall_5at100),

        # Final misc stuff
        @sub(subdf, preselection_iterations),
        @sub(subdf, preselection_oversample),
        @sub(subdf, lloyds_iterations),
    )

    temp = DataFrame(d)
    for row in eachrow(temp)
        push!(dest, row; cols = :union)
    end
    return nothing
end

#####
##### Quantized Query
#####

format_quantized_query(record::Record) = format_quantized_query(record.df)
function format_quantized_query(df::DataFrame)
    num_neighbors = unique(df[:, :num_neighbors])
    target_recalls = unique(df[:, :target_recall])

    dest = DataFrame()
    for nn in num_neighbors, target_recall in target_recalls
        _format_quantized_query!(dest, df, nn, target_recall)
    end
    return dest
end

function _format_quantized_query!(
    dest::DataFrame, src::DataFrame, nn::Integer, target_recall::Number
)
    # Filter so we only get the matching items.
    subdf = @from i in src begin
        @where i.num_neighbors == nn && i.target_recall == target_recall
        @select i
        @collect DataFrame
    end

    d = OrderedDict(
        @sub(subdf, num_neighbors),
        @sub(subdf, target_recall),
        @sub(subdf, maxlines),
        @sub(subdf, recall),
        @sub(subdf, qps),
        @sub(subdf, encoded_data_allocator),
        @sub(subdf, pqgraph_allocator),
        @sub(subdf, distance_type),
        @sub(subdf, distance_strategy),
        @sub(subdf, num_partitions),
        @sub(subdf, num_centroids),
        @sub(subdf, mean_latency),
        :latency_9 => getindex.(subdf[:, :nines_latency], 1),
        :latency_99 => getindex.(subdf[:, :nines_latency], 2),
        :latency_999 => getindex.(subdf[:, :nines_latency], 3),
        :latency_9999 => getindex.(subdf[:, :nines_latency], 4),
        @sub(subdf, max_latency),
        @sub(subdf, windowsize),
        @sub(subdf, data_allocator),
        @sub(subdf, graph_allocator),
    )

    temp = DataFrame(d)
    for row in eachrow(temp)
        push!(dest, row; cols = :union)
    end
    return nothing
end
