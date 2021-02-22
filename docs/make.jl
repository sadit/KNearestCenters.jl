using KNearestCenters
using Documenter

makedocs(;
    modules=[KNearestCenters],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/KNearestCenters.jl/blob/{commit}{path}#L{line}",
    sitename="KNearestCenters.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sadit.github.io/KNearestCenters.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sadit/KNearestCenters.jl",
    devbranch="main",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#"]
)
