using Documenter, CCBlade

makedocs(
    modules = [CCBlade],
    format = Documenter.HTML(),
    pages = [
        "Intro" => "index.md",
        "Guide" => "tutorial.md"
    ],
    repo="https://github.com/byuflowlab/CCBlade.jl/blob/{commit}{path}#L{line}",
    sitename="CCBlade.jl",
    authors="Andrew Ning <aning@byu.edu>",
)

deploydocs(
    repo = "github.com/byuflowlab/CCBlade.jl.git"
)