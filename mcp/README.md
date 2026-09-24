# CCBlade MCP server

A [Model Context Protocol](https://modelcontextprotocol.io) server that lets an LLM
client (Claude Desktop, Claude Code, ChatGPT, the MCP Inspector, ...) analyze, plot,
optimize and export propellers, hovering rotors and wind turbines with CCBlade by
calling typed tools instead of writing Julia.

The server is a separate Julia environment in this folder. CCBlade itself gains no new
dependencies: `mcp/Project.toml` depends on the parent package by path (`..`) plus
[ModelContextProtocol.jl](https://github.com/JuliaSMLM/ModelContextProtocol.jl),
[SNOW.jl](https://github.com/byuflowlab/SNOW.jl) (with the open-source Ipopt solver,
installed automatically), Plots, WriteVTK, FLOWMath and JSON3. Nothing has to be
downloaded by hand: `Pkg.instantiate()` fetches everything, including the Ipopt binary.

```
mcp/
├── Project.toml / Manifest.toml   the environment (Manifest is gitignored repo-wide; see below)
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
├── smoke_test.sh                  drives every tool over stdio with raw JSON-RPC, no LLM needed
├── test/run_lib_tests.jl          direct tests of RotorTools against the CCBlade docs values
├── claude_desktop_config.example.json
└── output/                        images, VTK files and ipopt.out land here (gitignored)
```

## Tools

| Tool | What it does |
|---|---|
| `list_presets` | The built-in rotors: `apc_10x5` propeller, `nasa_hover_rotor`, `nrel_5mw` wind turbine, each with a default operating point. |
| `list_airfoils` | Polar files in `CCBlade.jl/data` with Re, angle range, units and which presets use them. |
| `analyze_rotor` | One operating point: thrust, torque, power and the convention's coefficients (efficiency/CT/CQ/CP and J for propellers, figure of merit for hover, CP/CT and tip speed ratio for turbines); optional spanwise table. |
| `sweep_rotor` | One row of metrics per value of rpm, freestream, advance ratio, tip speed ratio or collective pitch, plus the best point. |
| `convert_rotor_units` | Thrust/CT, torque/CQ, power/CP, rpm/omega/rev-per-second/tip speed, freestream/J/tip speed ratio, efficiency, figure of merit, tip Mach, with the formulas used. |
| `plot_geometry` | Chord and twist distributions and the planform, as a PNG. |
| `plot_performance` | The sweep above as curves: efficiency (or FM or CP), force coefficients, thrust and power. |
| `plot_spanwise` | Loads, angle of attack and inflow angle, cl and cd, induction factors along the blade. |
| `plot_airfoil` | cl and cd versus alpha and the drag polar for a bundled airfoil file. |
| `optimize_rotor` | SNOW/Ipopt optimization of chord and twist at spanwise control points (and optionally rpm) with ForwardDiff derivatives through CCBlade: min power or max efficiency at a required thrust, max thrust at a power limit, or max figure of merit in hover. Returns before/after metrics, the optimized geometry arrays and a comparison image. |
| `export_blade_vtk` | Multiblock VTK file of all blades plus hub, lofted from a NACA 4-digit section, with chord, twist and spanwise loads attached, ready for ParaView. |

All inputs and outputs are SI; angles at the interface are degrees. Every geometry
field is optional: omit them all and you get the APC 10x5 at 5 m/s and 5400 rpm, name
a preset to start from it, or pass `r_over_R`, `chord_over_R` and `twist_deg` for a
custom blade. Operating-point fields you omit fall back to the preset's defaults and
are reported back under `operating_point_defaults_used`. Optional airfoil corrections
(Prandtl-Glauert Mach, Du-Selig/Eggers rotation, tip-loss model) are exposed as flags.

Every image a tool returns (the four `plot_*` tools and `optimize_rotor`) is also written
to `mcp/output/<preset>_<plot>_<timestamp>.png`, and the path comes back as `image_file`
in the JSON, so the figure is on disk even when the client doesn't display images.

## Setup

Requires Julia 1.10 or newer and about 3 GB of disk for packages. From the repository
root:

```bash
git clone https://github.com/byuflowlab/CCBlade.jl.git
cd CCBlade.jl
git checkout mcp-demo
./mcp/setup.sh          # or: JULIA=/path/to/julia ./mcp/setup.sh
```

`setup.sh` installs and precompiles the packages and then runs a self-check. **It takes
5-15 minutes on a first run** -- almost entirely precompilation, so do it before a
workshop, not during one.

`Manifest.toml` is not tracked, so each clone resolves its own package versions within
the compat bounds in `mcp/Project.toml`. A fresh resolve was verified to reproduce the
same results as the development machine.

The equivalent by hand, if you prefer:

```bash
julia --startup-file=no --project=mcp -e 'using Pkg; Pkg.instantiate()'
```

Check the library directly (compares against numbers from the CCBlade docs) and the
notebook's cell logic:

```bash
julia --startup-file=no --project=mcp mcp/test/run_lib_tests.jl
julia --startup-file=no --project=mcp mcp/test/run_notebook_tests.jl
```

Check the whole server end to end without an LLM (the first run compiles for a minute
or two; images are written to `mcp/output/smoke_*.png`):

```bash
./mcp/smoke_test.sh
# or, if `julia` on your PATH is not the one you want:
JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia ./mcp/smoke_test.sh
```

## Connect a client

The stdio command is the same everywhere. Use absolute paths, and keep
`--startup-file=no` so nothing from your `startup.jl` runs in the server process.
(`server.jl` also reserves the real stdout for the protocol and sends every other
print to stderr, so Ipopt or plotting output cannot corrupt the stream.)

Print the exact command and config for your own machine, with absolute paths already
filled in:

```bash
./mcp/print_client_config.sh
```

**Claude Desktop.** Settings, Developer, Edit Config, then merge in the JSON that
`print_client_config.sh` prints (or `claude_desktop_config.example.json` with the paths
filled in by hand). Restart Claude Desktop; the tools appear under the tools icon in a
new chat.

Point it at a real Julia binary, not a shell alias or a `juliaup` shim: MCP clients do
not load your shell profile, and `juliaup`'s launcher may resolve to a different Julia
version than your interactive shell does.

**Claude Code.**

```bash
claude mcp add ccblade -- /path/to/julia --startup-file=no \
    --project=/ABSOLUTE/PATH/TO/CCBlade.jl/mcp /ABSOLUTE/PATH/TO/CCBlade.jl/mcp/server.jl
```

**MCP Inspector** (a browser UI that shows every request and response; good for
teaching what the protocol looks like):

```bash
npx @modelcontextprotocol/inspector /path/to/julia --startup-file=no --project=mcp mcp/server.jl
```

**Remote / HTTP.** For clients that connect over the network (Claude.ai custom
connectors, the OpenAI Responses API, ChatGPT developer mode) run the streamable HTTP
transport and put it behind HTTPS:

```bash
julia --startup-file=no --project=mcp mcp/server.jl --http 3000          # 127.0.0.1:3000
julia --startup-file=no --project=mcp mcp/server.jl --http 3000 0.0.0.0  # all interfaces
```

Anything reachable from the internet should be authenticated; ModelContextProtocol.jl
supports bearer-token and OAuth resource-server validation through the `auth` keyword
of `HttpTransport`, see its README.

Set `CCBLADE_MCP_WARMUP=0` in the environment to skip the start-up warm-up run.

## Before a demo

One command runs every check (library values, notebook logic, a headless Pluto run with
all branches exercised, and the whole server over raw JSON-RPC):

```bash
JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia ./mcp/check_all.sh
```

It prints `ALL CHECKS PASSED - ready to demo.` or names what broke. Takes a couple of
minutes, almost all of it Julia compilation. Run it the morning of the talk: the usual
cause of a failure is an unrelated package update changing the environment.

Also worth doing beforehand, because each takes a minute of first-run compilation that
is dull to watch live:

- Open the notebook once and move a slider, so Plots and CCBlade are compiled.
- Start a chat with the MCP client and ask one throwaway question, so the server is warm.
- Have `mcp/output/` already populated (`check_all.sh` does this) and a ParaView window
  open with a previously exported `.vtm`, so the 3D view is one file-open away.

## Demo prompts

- "Analyze the default propeller at 10 m/s and 6000 rpm and tell me the efficiency."
- "Plot efficiency versus advance ratio from 0.1 to 0.8 at 5400 rpm. Where does it peak?"
- "Same rotor with 3 blades, how does peak efficiency change?"
- "Show me the spanwise loads and angle of attack at the peak efficiency point."
- "Redesign the chord and twist to minimize power while producing at least 3 N of
  thrust at 10 m/s and 5400 rpm. Then plot the new geometry."
- "Design a 0.3 m radius, 4-blade hover rotor with linear taper from 0.12 R to 0.06 R
  and twist from 25 to 10 degrees. What figure of merit does it get at 3000 rpm?
  Optimize it for maximum figure of merit at 40 N of thrust."
- "Plot the CP versus tip speed ratio curve of the NREL 5 MW turbine at 10 m/s."
- "I measured 4.2 N of thrust and 45 W at 6000 rpm on a 10 inch prop at 8 m/s. What
  are CT, CP, J and efficiency?"
- "Export the optimized blade for ParaView."

## Interactive notebook

`notebook/rotor_explorer.jl` is a Pluto notebook over the same `RotorTools` module the
server calls. Move a slider and every plot re-solves; there is no MCP involved.

```bash
./mcp/notebook/run_notebook.sh
# or
JULIA=/Applications/Julia-1.10.app/Contents/Resources/julia/bin/julia ./mcp/notebook/run_notebook.sh
```

Pluto prints a `localhost` URL with a secret token. The first run compiles CCBlade,
Plots and Ipopt, which takes a minute or two; after that a slider move redraws in well
under a second.

What it exposes:

- **Preset** picker (propeller / hover rotor / wind turbine). Slider ranges and the
  operating-point defaults follow the preset.
- **Operating point**: rpm, freestream, collective pitch, air density.
- **Geometry**: blade count, tip radius, and scale factors on the preset's own chord and
  twist distributions plus a twist offset, so you can walk a family of blades without
  typing arrays.
- **Corrections**: tip/hub loss model, Mach, rotational stall delay.
- **Live output**: a performance table in the rotor type's own convention, the geometry
  plot, and the spanwise loads/angles/coefficients/inductions plot.
- **Button-gated** (too slow for every keystroke): the performance sweep, a SNOW/Ipopt
  optimization with its before/after plot, and the ParaView export.
- **Handoff**: the last cell prints the current design as JSON in exactly the shape the
  MCP tools accept, so a design found with sliders can be pasted into a conversation
  with Claude and taken further there.

Two Pluto gotchas worth knowing if you edit it:

- Inside a multi-line `md"""` block, interpolate **bare variables only**. `$(f(x))` and
  `$(d["k"])` can be mis-parsed, and nothing is interpolated inside backtick code spans.
  Compute the value in a preceding line and interpolate that. The `error_box` helper in
  the setup cell exists for the same reason.
- A cell that must not rerun on every slider move takes a `CounterButton` binding and
  references it as a bare statement, which is what makes the sweep and optimizer opt-in.

`test/run_notebook_tests.jl` re-runs the computational body of each cell headlessly, so
a refactor of `lib/` that breaks the notebook fails in CI rather than in front of an
audience.

## Extending

- The physics lives in `lib/`. To add a tool, write a plain function there (raise
  `fail("message")` for bad input), then add an `*_impl(params)` and an `MCPTool` entry
  with a JSON schema in `tools.jl`. Handlers return a `Dict` (serialized to JSON) or a
  vector of `TextContent` / `ImageContent` blocks.
- Arrays and enums need a raw JSON schema through `input_schema`; the simpler
  `ToolParameter` API only covers flat string/number/boolean arguments.
- Long-running tools (a big sweep, a large optimization) should take a `(params, ctx)`
  handler and call `send_progress(ctx, ...)`, or set `task_support = :optional` and
  `task_detach(ctx)` to run in the background. See the ModelContextProtocol.jl docs.
- `RotorTools` is deliberately MCP-free so an interactive Pluto notebook or a script can
  drive the same presets, analysis, plots and optimizer. The notebook below is that
  second front end.

## Running the training

A suggested 60-minute shape, building from "what is this" to "here is how you would add
one". The ordering is deliberate: show the problem before the protocol, and the protocol
before the code.

**1. The problem, without any AI (5 min).** Put `test/test168.csv`-style input up, or
just the CCBlade tutorial script. Point out that using this code today means knowing
Julia, knowing the struct layout, and knowing which function to call. That is the
barrier the MCP removes. Do not mention MCP yet.

**2. The payoff first (10 min).** Open the MCP client and ask, in plain English:
"Analyze the default propeller at 10 m/s and 6000 rpm and tell me the efficiency."
Then "plot efficiency versus advance ratio and tell me where it peaks." Let the room see
a plot appear from a sentence. Only then say: the model did not know anything about
CCBlade; it read a list of tools at connection time.

**3. What actually crossed the wire (10 min).** This is the part that makes MCP click,
and it is where most talks skip straight to code. Run the MCP Inspector, or show the
output of `./mcp/smoke_test.sh`, and walk through three messages: `initialize`,
`tools/list`, and one `tools/call` with its JSON result. Emphasize that this is ordinary
JSON-RPC over stdin/stdout, that the tool list is just data, and that nothing in the
server is model-specific -- the same server works from Claude Desktop, Claude Code, or
the OpenAI Responses API.

**4. How a tool is defined (10 min).** Show one `MCPTool` in `tools.jl` next to the
plain function it wraps in `lib/`. The message to land: the description field is the
prompt -- it is how the model knows when to call this -- and the schema is the contract.
Then show `guarded` turning a `ToolError` into a readable message, and note that a good
error message is what lets the model retry correctly instead of guessing.

**5. Something real (10 min).** Run the optimization: "Redesign the chord and twist to
minimize power while producing at least 3 N of thrust at 10 m/s and 5400 rpm, then plot
the new geometry." It takes ~15 s and returns `Solve_Succeeded` with a before/after
plot. This is the slide that convinces people it is not a toy: a gradient-based
optimization with ForwardDiff derivatives through the solver, driven by a sentence.
Then export to ParaView and show the blade.

**6. The other front end (10 min).** Open the Pluto notebook and move the sliders.
Say explicitly: this is the *same module*, not a reimplementation. `lib/RotorTools.jl`
knows nothing about MCP; `tools.jl` is a thin wrapper, and the notebook is a second
wrapper. That separation is the transferable lesson -- keep your science in plain
functions and both an LLM and a human GUI can drive it. Finish with the handoff cell:
copy the JSON out of the notebook into the chat, and continue there.

**7. How they would start (5 min).** `Pkg.instantiate()`, point a client at
`server.jl`, write one function and one `MCPTool`. Note the cloud story briefly
(`--http`, needs HTTPS and auth) and that it is a bigger lift than a local demo.

Practical notes:

- **Have a fallback.** Keep a terminal with `smoke_test.sh` output already scrolled, and
  the PNGs in `mcp/output/` open. If the network or the client misbehaves mid-demo you
  can keep talking from artifacts rather than debugging in front of people.
- **The optimization is the best demo and the slowest.** Start it, then talk about what
  Ipopt is doing while it runs, rather than watching a spinner in silence.
- **Expect the "can it hallucinate a wrong answer" question.** The honest answer: the
  model cannot invent numbers, because the numbers come from CCBlade -- but it can call
  the wrong tool or pass a wrong value, which is why the schemas constrain inputs and
  why errors are written to be readable. Show the deliberate bad-airfoil call at the end
  of `smoke_test.sh`.
- **Expect "does this send my data to Anthropic?"** For the stdio server: the tool
  definitions and the arguments and results of calls go to the model; the code and files
  do not, and the computation runs locally. Worth being precise about for lab work.

## Reproducibility

`Manifest.toml` is deliberately not tracked (the repository's root `.gitignore` excludes
every `Manifest.toml`). Each clone resolves its own versions, constrained by the compat
bounds in `mcp/Project.toml`, which pin every direct dependency to the major version
this was tested against.

A fresh clone with no manifest was verified to reproduce the development machine's
numbers exactly -- APC thrust 3.13288 N, NREL CP 0.469672, `min_power` converging in 52
evaluations. If a future resolve ever disagrees, `mcp/test/run_lib_tests.jl` compares
against the CCBlade documentation values and will say so.
