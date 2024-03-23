using Conflux
using Documenter

DocMeta.setdocmeta!(Conflux, :DocTestSetup, :(using Conflux); recursive=true)

makedocs(;
    modules=[Conflux],
    authors="Anton Oresten <anton.oresten42@gmail.com>",
    sitename="Conflux.jl",
    doctest=false,
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "API.md",
    ],
    checkdocs=:all,
)

deploydocs(
    repo = "github.com/MurrellGroup/Conflux.jl.git",
    devbranch = "main",
)