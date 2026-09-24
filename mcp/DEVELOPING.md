# Developing the CCBlade MCP server

The server is a separate Julia environment in this folder. CCBlade itself gains no new
dependencies: `mcp/Project.toml` depends on the parent package by path (`..`) plus
[ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl),
[SNOW.jl](https://github.com/byuflowlab/SNOW.jl) (with Ipopt, installed automatically),
Plots, WriteVTK, FLOWMath and JSON3.

## Layout

```
mcp/
├── Project.toml                   the environment (Manifest.toml is gitignored repo-wide)
├── lib/RotorTools.jl              plain-Julia layer: presets, geometry, analysis, units, plots,
│   ├── presets.jl                 optimization, VTK export. Knows nothing about MCP, so a
│   ├── geometry.jl                notebook or script can reuse it unchanged.
│   ├── analysis.jl
│   ├── units.jl
│   ├── plotting.jl
│   ├── optimize.jl
│   └── vtk.jl
├── tools.jl                       MCP layer: JSON schemas + thin handlers over RotorTools
├── server.jl                      entry point: warm-up, then serve over stdio or HTTP
├── notebook/rotor_explorer.jl     Pluto notebook over the same RotorTools module
├── test_server.sh                  drives every tool over stdio with raw JSON-RPC, no LLM needed
├── check_all.sh                   runs every test and check below
├── test/                          library, notebook and headless-Pluto tests
└── output/                        images, VTK files and ipopt.out land here (gitignored)
```

## Tests

```bash
julia --startup-file=no --project=mcp mcp/test/run_lib_tests.jl       # vs. CCBlade docs values
julia --startup-file=no --project=mcp mcp/test/run_notebook_tests.jl  # notebook cell logic
./mcp/test_server.sh                                                    # whole server over stdio
./mcp/check_all.sh                                                     # all of the above
```

Prefix any of these with `JULIA=/path/to/julia` if `julia` on your PATH isn't the one
you want.

## Tool behavior

- Every geometry field is optional. With none, you get the APC 10x5 at 5 m/s and 5400
  rpm. Name a preset to start from it, or pass `r_over_R`, `chord_over_R` and
  `twist_deg` for a custom blade. Operating-point fields you omit fall back to the
  preset's defaults and are reported under `operating_point_defaults_used`.
- Optional corrections (Prandtl-Glauert Mach, Du-Selig/Eggers rotation, tip-loss model)
  are exposed as flags.
- Every image a tool returns (the four `plot_*` tools and `optimize_rotor`) is also
  written to `mcp/output/<preset>_<plot>_<timestamp>.png`, and its path is returned as
  `image_file`.
- `server.jl` reserves the real stdout for the protocol and sends every other print to
  stderr, so Ipopt or plotting output can't corrupt the stream. Keep `--startup-file=no`
  so nothing from your `startup.jl` runs in the server process.
- Set `CCBLADE_MCP_WARMUP=0` to skip the start-up warm-up run.

## Adding a tool

- The physics lives in `lib/`. Write a plain function there (raise `fail("message")` for
  bad input), then add an `*_impl(params)` and an `MCPTool` entry with a JSON schema in
  `tools.jl`. Handlers return a `Dict` (serialized to JSON) or a vector of
  `TextContent` / `ImageContent` blocks.
- Arrays and enums need a raw JSON schema through `input_schema`; the simpler
  `ToolParameter` API only covers flat string/number/boolean arguments.
- Long-running tools should take a `(params, ctx)` handler and call
  `send_progress(ctx, ...)`, or set `task_support = :optional` and `task_detach(ctx)` to
  run in the background. See the ModelContextProtocol.jl docs.

## Other clients

**MCP Inspector**, a browser UI that shows every request and response:

```bash
npx @modelcontextprotocol/inspector /path/to/julia --startup-file=no --project=mcp mcp/server.jl
```

**HTTP.** For clients that connect over the network (Claude.ai custom connectors, the
OpenAI Responses API, ChatGPT developer mode), run the streamable HTTP transport and
put it behind HTTPS:

```bash
julia --startup-file=no --project=mcp mcp/server.jl --http 3000          # 127.0.0.1:3000
julia --startup-file=no --project=mcp mcp/server.jl --http 3000 0.0.0.0  # all interfaces
```

Anything reachable from the internet should be authenticated. ModelContextProtocol.jl
supports bearer-token and OAuth validation through the `auth` keyword of
`HttpTransport`; see its README.

## Pluto notebook

`notebook/rotor_explorer.jl` drives the same `RotorTools` module with sliders, without
MCP:

```bash
./mcp/notebook/run_notebook.sh
```

It has a preset picker, operating-point, geometry and correction sliders, live
performance, geometry and spanwise plots, and button-gated sweep, optimization and
ParaView export. The last cell prints the current design as JSON in the shape the MCP
tools accept, so a design found with sliders can be pasted into a chat.

Two Pluto gotchas if you edit it:

- Inside a multi-line `md"""` block, interpolate bare variables only. `$(f(x))` and
  `$(d["k"])` can be mis-parsed, and nothing is interpolated inside backtick code spans.
  The `error_box` helper in the setup cell exists for the same reason.
- A cell that must not rerun on every slider move takes a `CounterButton` binding and
  references it as a bare statement.

## Reproducibility

`Manifest.toml` is not tracked, so each clone resolves its own versions within the
compat bounds in `mcp/Project.toml`. A fresh clone was verified to reproduce the
development machine's numbers exactly (APC thrust 3.13288 N, NREL CP 0.469672,
`min_power` converging in 52 evaluations). If a future resolve disagrees,
`test/run_lib_tests.jl` will say so.
