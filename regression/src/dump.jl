format_query(record::Record) = format_query(record.df)
function format_query(df::DataFrame)
    # Loop through num neighbors and recall.
    num_neighbors = unique(df[:, :num_neighbors])
    target_recalls = unique(df[:, :target_recall])

    dest = DataFrame()
    for nn in num_neighbors, target_recall in target_recalls
        @show nn
        @show target_recall
        _format_query!(dest, df, nn, target_recall)
    end
    return dest
end

macro sub(df, sym)
    sym = QuoteNode(sym)
    return :($sym => $(esc(df))[:, $sym])
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
