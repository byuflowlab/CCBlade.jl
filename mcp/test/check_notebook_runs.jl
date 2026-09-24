# Open the Pluto notebook headlessly, run every cell and report any that errored.
#
# This catches what run_notebook_tests.jl cannot: Pluto-specific problems such as a
# markdown interpolation that fails to parse, or a cell whose output errors even though
# the underlying computation is fine. Run it before a demo.
#
#   julia --startup-file=no --project=mcp mcp/test/check_notebook_runs.jl
#   julia --startup-file=no --project=mcp mcp/test/check_notebook_runs.jl --buttons
#
# With --buttons the CounterButton cells are rewritten to "pressed" first, so the sweep,
# optimization and VTK branches run too (slower, about a minute).

using Pluto

const NOTEBOOK = normpath(joinpath(@__DIR__, "..", "notebook", "rotor_explorer.jl"))

"Rewrite the CounterButton bindings so the button-gated cells execute."
function pressed_copy(src)
    code = read(src, String)
    for sym in ("run_sweep", "run_opt", "run_vtk")
        code = replace(code, Regex("@bind $sym CounterButton\\(\"[^\"]*\"\\)") => "$sym = 1; nothing")
    end
    # The copy must sit next to the original: the setup cell activates the environment
    # with Pkg.activate(joinpath(@__DIR__, "..")).
    tmp = joinpath(dirname(src), "_check_buttons_tmp.jl")
    write(tmp, code)
    return tmp
end

function main()
    buttons = "--buttons" in ARGS
    path = buttons ? pressed_copy(NOTEBOOK) : NOTEBOOK
    try
        session = Pluto.ServerSession()
        session.options.evaluation.workspace_use_distributed = false
        nb = Pluto.SessionActions.open(session, path; run_async = false)

        bad = 0
        for c in nb.cells
            c.errored || continue
            bad += 1
            println("ERRORED CELL: ", first(strip(c.code), 140))
            println("   -> ", first(replace(string(c.output.body), "\n" => " "), 300), "\n")
        end
        println("cells: $(length(nb.cells)), errored: $bad", buttons ? " (buttons pressed)" : "")
        return bad
    finally
        # Cleanup must happen before exit(): exit() inside a try skips the finally block.
        buttons && rm(path; force = true)
    end
end

exit(main() == 0 ? 0 : 1)

