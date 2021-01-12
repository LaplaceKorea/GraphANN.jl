Base.@kwdef mutable struct Clustering
    partition_size::Int
    num_centroids::Int
    preselection_iterations::Int = 3
    preselection_oversample::Float64 = 1.5
    lloyds_iterations::Int = 5
end

function computerecall(groundtruth, ids, n::Int, recall_at::Int)
    vgt = view(groundtruth, 1:n, :)
    vids = view(ids, 1:recall_at, :)

    matches = sum(length.(intersect.(eachcol(vgt), eachcol(vids))))
    return matches / (n * size(groundtruth, 2))
end

function cluster(record, dataset::Dataset, clustering::Clustering; saveprefix = nothing)
    data = load(dataset)
    stats_pre = @elapsed precentroids = GraphANN._Quantization.choose_centroids(
        data,
        clustering.partition_size,
        clustering.num_centroids;
        num_iterations = clustering.preselection_iterations,
        oversample = clustering.preselection_oversample,
    )

    stats_refine = @elapsed refinedcentroids = GraphANN._Quantization.refine(
        precentroids,
        data,
        clustering.num_centroids,
    )

    stats_lloyds = @elapsed finalcentroids = GraphANN._Quantization.lloyds(
        refinedcentroids,
        data;
        num_iterations = clustering.lloyds_iterations,
    )

    if saveprefix !== nothing
        savename = "pq_$(clustering.partition_size)x$(clustering.num_centroids).jls"
        savepath = joinpath(SCRATCH, saveprefix, savename)
        serialize(savepath, finalcentroids)
    else
        savename = nothing
        savepath = nothing
    end

    # Compute the groundtruth for this clustering.
    pqtable = GraphANN.PQTable{size(finalcentroids, 2)}(finalcentroids)
    encoder = GraphANN._Quantization.binned(pqtable)
    encode_type = clustering.num_centroids <= 256 ? UInt8 : UInt16
    encode_time = @elapsed data_encoded = GraphANN._Quantization.encode(encode_type, encoder, data)

    queries = load_queries(dataset)
    groundtruth = load_groundtruth(dataset)

    # Use the PQTable as the metric since it knows how to decode the encoded dataset.
    gttime = @elapsed gt_encoded = 1 .+ GraphANN.bruteforce_search(
        queries,
        data_encoded,
        100;
        metric = pqtable
    )

    results = makeresult([
        dict(dataset),
        dict(clustering),
        Dict(
            :clustering => Dict(
                :preselection => dict(stats_pre; excluded = :value),
                :refinement => dict(stats_refine; excluded = :value),
                :lloyds => dict(stats_lloyds; excluded = :value),
            ),
            :encode_time => encode_time,
            :groundtruth_time => gttime,
            :savename => savename,
            :saveprefix => saveprefix,
            :savepath => savepath,
            :num_threads => Threads.nthreads(),

            # Recalls
            :recall_1at1 => computerecall(groundtruth, gt_encoded, 1, 1),
            :recall_1at5 => computerecall(groundtruth, gt_encoded, 1, 5),
            :recall_1at10 => computerecall(groundtruth, gt_encoded, 1, 10),
            :recall_1at20 => computerecall(groundtruth, gt_encoded, 1, 20),
            :recall_1at100 => computerecall(groundtruth, gt_encoded, 1, 100),

            :recall_5at5 => computerecall(groundtruth, gt_encoded, 5, 5),
            :recall_5at10 => computerecall(groundtruth, gt_encoded, 5, 10),
            :recall_5at20 => computerecall(groundtruth, gt_encoded, 5, 20),
            :recall_5at100 => computerecall(groundtruth, gt_encoded, 5, 100),

            :recall_10at10 => computerecall(groundtruth, gt_encoded, 10, 10),
            :recall_10at20 => computerecall(groundtruth, gt_encoded, 10, 20),
            :recall_10at100 => computerecall(groundtruth, gt_encoded, 10, 100),
        )
    ])
    push!(record, results)
    save(record)
end
