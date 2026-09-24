"""
    RotorTools

Plain-Julia layer on top of CCBlade used by the MCP server (`mcp/tools.jl`) and
reusable from a notebook or script. It provides built-in rotor presets, resolution of
geometry and operating points from plain dictionaries, analysis and sweeps, unit
conversion, plotting to PNG bytes, SNOW/Ipopt optimization and ParaView export.

It knows nothing about MCP: every function takes ordinary Julia values or a
`Dict{String,Any}` of JSON-like inputs and returns Julia values, so a Pluto notebook or
a test script can drive exactly the same code an LLM client does.
"""
module RotorTools

using CCBlade
using FLOWMath: akima
using SNOW
using WriteVTK

# GR (the Plots backend) must be told it is headless before Plots loads, otherwise it
# may try to open a window or write to stdout.
get!(ENV, "GKSwstype", "100")
using Plots

const CCBLADE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const AIRFOIL_DIR = joinpath(CCBLADE_ROOT, "data")
const OUTPUT_DIR = normpath(joinpath(@__DIR__, "..", "output"))

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

"Raised for bad user input; the MCP layer turns it into a readable error message."
struct ToolError <: Exception
    msg::String
end
fail(msg) = throw(ToolError(msg))

"Round for output; non-finite values become `nothing` so they serialize as JSON null."
sig(x::Real, n=6) = isfinite(x) ? round(Float64(x); sigdigits=n) : nothing
sig(x, n=6) = x

present(p, key) = haskey(p, key) && p[key] !== nothing
getnum(p, key, default) = present(p, key) ? Float64(p[key]) : default
getint(p, key, default) = present(p, key) ? Int(p[key]) : default
getstr(p, key, default) = present(p, key) ? String(p[key]) : default
getbool(p, key, default) = present(p, key) ? Bool(p[key]) : default
getvec(p, key) = present(p, key) ? Float64.(collect(p[key])) : nothing
getstrvec(p, key) = present(p, key) ? String.(collect(p[key])) : nothing

"Extract a field from a vector of CCBlade `Outputs` (works for StructArrays and plain vectors)."
field(outputs, s::Symbol) = [getproperty(o, s) for o in outputs]

"Trapezoidal integral, used for solidity."
trapz(x, y) = sum((x[i+1] - x[i]) * (y[i+1] + y[i]) / 2 for i in 1:length(x)-1)

include("presets.jl")
include("geometry.jl")
include("analysis.jl")
include("units.jl")
include("plotting.jl")
include("optimize.jl")
include("vtk.jl")

export ToolError, fail, sig, present, getnum, getint, getstr, getbool, getvec, getstrvec, field,
       CCBLADE_ROOT, AIRFOIL_DIR, OUTPUT_DIR,
       PRESETS, preset_summaries, list_airfoils, load_airfoil,
       Geometry, resolve_geometry, rebuild_geometry, geometry_info, geometry_arrays,
       OperatingPoint, resolve_operating, with_rpm, evaluate, section_table, sweep,
       SWEEP_VARIABLES, best_metric_key, best_point,
       CONVENTIONS, convert_units,
       plot_geometry_png, plot_performance_png, plot_spanwise_png, plot_airfoil_png, plot_optimization_png,
       OBJECTIVES, DESIGN_VARIABLE_SETS, optimize_rotor,
       export_blade_vtk

end # module
