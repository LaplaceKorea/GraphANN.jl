# -- Index Building top level
function index_building(
    record::Record,
    dataset::Dataset,
    parameters::GraphANN.GraphParameters;
    savepath = nothing,
    batchsize = 50000,
    allocator = GraphANN.stdallocator,
)
    data = load(dataset)
    stats = @timed meta = GraphANN.generate_index(data, parameters; batchsize = batchsize)

    if savepath != nothing
        GraphANN.save(savepath, meta.graph)
    end

    # Build up the things we want to save
    results = makeresult([
        dict(stats; excluded = :value),
        dict(dataset),
        dict(parameters),
        Dict(
            :batchsize => batchsize,
            :savepath => savepath,
            :num_threads => Threads.nthreads(),
            :graph_allocator => allocator,
        )
    ])

    # Save the results!
    push!(record, results)
    save(record)
    return nothing
end
