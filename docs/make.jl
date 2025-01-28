using Documenter, CCBlade

makedocs(
    modules = [CCBlade],
    format = Documenter.HTML(),
    pages = [
        "Intro" => "index.md",
        "Quick Start" => "tutorial.md",
        "Guided Examples" => "howto.md",
        "API Reference" => "reference.md",
        "Theory" => "theory.md"
    ],
    repo="https://github.com/byuflowlab/CCBlade.jl/blob/{commit}{path}#L{line}",
    sitename="CCBlade.jl",
    authors="Andrew Ning <aning@byu.edu>",
    warnonly = Documenter.except(:linkcheck, :footnote),
)

deploydocs(
    repo = "github.com/byuflowlab/CCBlade.jl.git"
)