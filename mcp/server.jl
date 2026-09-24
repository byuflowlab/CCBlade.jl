#!/usr/bin/env julia
# CCBlade MCP server.
#
#   julia --startup-file=no --project=mcp mcp/server.jl              # stdio (Claude Desktop, Claude Code, MCP Inspector)
#   julia --startup-file=no --project=mcp mcp/server.jl --http 3000  # streamable HTTP on http://127.0.0.1:3000/
#
# In stdio mode stdout *is* the protocol channel: a single stray print (from Ipopt, a
# plotting backend, or a debugging println) would corrupt it. So the first thing this
# script does is duplicate the real stdout for the protocol and point everything else
# at stderr, where the client shows it as server logs.

using ModelContextProtocol

const PROTOCOL_OUT = Base.fdio(Int(Base.cconvert(Cint, Base.Libc.dup(RawFD(1)))), true)
redirect_stdout(stderr)

include(joinpath(@__DIR__, "tools.jl"))

const SERVER_INSTRUCTIONS = """
CCBlade is a blade element momentum solver for propellers, hovering rotors and wind
turbines. All inputs and outputs are SI (meters, m/s, rpm, N, N*m, W); angles at this
interface are in degrees. Start from a preset (list_presets) or give r_over_R,
chord_over_R and twist_deg for a custom blade; operating-point fields you omit fall
back to the preset's defaults and are reported. Use analyze_rotor for one point,
sweep_rotor / plot_performance for curves, plot_spanwise for load distributions,
optimize_rotor to redesign chord and twist for a thrust or power requirement,
convert_rotor_units to move between coefficients and dimensional values, and
export_blade_vtk to hand the user a ParaView file. For hover use Vinf_m_s = 0 with
rotor_type = "helicopter"; for wind turbines use rotor_type = "windturbine" with a
tip_speed_ratio or rpm.
"""

function build_server()
    return mcp_server(
        name = "ccblade",
        version = "0.2.0",
        title = "CCBlade rotor analysis and design",
        description = "Blade element momentum analysis, plotting and optimization of propellers, rotors and wind turbines with CCBlade.jl",
        instructions = SERVER_INSTRUCTIONS,
        tools = CCBLADE_TOOLS,
    )
end

# Compile the common paths once so the first tool calls answer quickly. Kept short so the
# client's initialize handshake is not delayed by more than a few seconds; the optimizer
# and VTK export compile on first use instead.
function warm_up()
    t = @elapsed begin
        analyze_rotor_impl(Dict{String,Any}("include_sections" => true))
        sweep_rotor_impl(Dict{String,Any}("sweep_variable" => "advance_ratio", "values" => [0.2, 0.4]))
        plot_geometry_impl(Dict{String,Any}())
        convert_units_impl(Dict{String,Any}("rpm" => 5400, "thrust_N" => 3.0, "Vinf_m_s" => 5.0))
    end
    @info "CCBlade warm-up complete" seconds = round(t; digits = 1)
end

get(ENV, "CCBLADE_MCP_WARMUP", "1") == "0" || warm_up()
server = build_server()

if !isempty(ARGS) && ARGS[1] == "--http"
    port = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3000
    host = length(ARGS) >= 3 ? ARGS[3] : "127.0.0.1"
    transport = HttpTransport(host = host, port = port)
    ModelContextProtocol.connect(transport)   # binds the socket; start! does not do this for HTTP
    @info "Serving MCP over streamable HTTP" url = "http://$host:$port/"
    start!(server; transport = transport)
else
    start!(server; transport = StdioTransport(input = stdin, output = PROTOCOL_OUT))
end
