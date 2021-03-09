using GraphANN
using Documenter

DocMeta.setdocmeta!(GraphANN, :DocTestSetup, :(using GraphANN); recursive=true)

makedocs(;
    modules=[GraphANN],
    authors="Hildebrand, Mark <mark.hildebrand@intel.com> and contributors",
    repo="https://gitlab.devtools.intel.com/stg-ai-sw-team/GraphANN.jl/-/blob/{commit}{path}#{line}",
    sitename="GraphANN.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Algorithms" => [
            "Exhaustive" => "exhaustive.md",
            "DiskANN" => "diskann.md",
        ],
        "Saving and Loading" => "io.md",
        "Internals" => [
            "Fundamentals" => "fundamentals.md",
            "Graphs" => "graphs.md",
            "Utilities" => "utilities.md",
        ],
    ],
)
