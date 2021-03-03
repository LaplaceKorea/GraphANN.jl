using GraphANN
using Documenter

DocMeta.setdocmeta!(GraphANN, :DocTestSetup, :(using GraphANN); recursive=true)

makedocs(;
    modules=[GraphANN],
    authors="Hildebrand, Mark <mark.hildebrand@intel.com> and contributors",
    repo="https://github.com/hildebrandmw/GraphANN.jl/blob/{commit}{path}#{line}",
    sitename="GraphANN.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Fundamentals" => "fundamentals.md",
    ],
)
