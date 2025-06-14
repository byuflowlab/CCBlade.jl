using CCBlade
using Documenter

DocMeta.setdocmeta!(CCBlade, :DocTestSetup, :(using CCBlade); recursive=true)

makedocs(
    modules = [CCBlade],
    authors="Andrew Ning <aning@byu.edu> and contributors",
    sitename="CCBlade.jl",
    format=Documenter.HTML(;
        canonical="https://flow.byu.edu/CCBlade.jl",
        edit_link="master",
        assets=String[],
    ),
    pages = [
        "Intro" => "index.md",
        "Quick Start" => "tutorial.md",
        "Guided Examples" => "howto.md",
        "API Reference" => "reference.md",
        "Theory" => "theory.md"
    ],
    warnonly = Documenter.except(:linkcheck, :footnote),
)

deploydocs(;
    repo="github.com/byuflowlab/CCBlade.jl",
    versions = ["stablle" => "master"],
    # devbranch="master",
)