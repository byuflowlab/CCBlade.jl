using Documenter, CCBlade

makedocs(
    format = Documenter.HTML(),
    sitename = "CCBlade Documentation",
    pages = [
        "index.md",
        "tutorial.md"
    ],
    modules = [CCBlade]
)


# deploydocs(
#     repo = "github.com/byuflowlab/CCBlade.jl.git",
#     julia = "0.6"
# )