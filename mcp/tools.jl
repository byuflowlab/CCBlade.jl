# MCP tool definitions for CCBlade. The physics lives in lib/RotorTools.jl; this file
# only declares JSON schemas, maps tool arguments onto RotorTools calls, and shapes the
# results (JSON text and PNG images) for the client.

using ModelContextProtocol
using JSON3
include(joinpath(@__DIR__, "lib", "RotorTools.jl"))
using .RotorTools

text(d) = TextContent(text = JSON3.write(d))
image(bytes) = ImageContent(data = bytes, mime_type = "image/png")

# Every plot is also written to OUTPUT_DIR so the user has a file to open; the path goes
# into the JSON as "image_file". Names are <label>_<tool>_<timestamp>.png.
function saved_plot(result::Dict, bytes, label, tool)
    mkpath(OUTPUT_DIR)
    stamp = Libc.strftime("%Y%m%d-%H%M%S", time()) * "-" * lpad(round(Int, 1000 * time()) % 1000, 3, '0')
    path = joinpath(OUTPUT_DIR, replace("$(label)_$(tool)_$(stamp)", r"[^A-Za-z0-9_.-]" => "_") * ".png")
    write(path, bytes)
    result["image_file"] = path
    return [text(result), image(bytes)]
end

# Handlers return a Dict (serialized to JSON) or a Vector of content blocks. Input
# problems come back as {"error": ...} so the model can correct its call; anything else
# is logged with a stack trace to stderr and reported briefly.
function guarded(f)
    return params -> try
        f(params)
    catch e
        if e isa ToolError
            Dict{String,Any}("error" => e.msg)
        else
            @error "tool call failed" exception = (e, catch_backtrace())
            Dict{String,Any}("error" => "Internal error: " * first(sprint(showerror, e), 500))
        end
    end
end

# ----------------------------------------------------------------------------
# Schema helpers
# ----------------------------------------------------------------------------

function prop(type, description; kwargs...)
    d = Dict{String,Any}("type" => type, "description" => description)
    for (k, v) in kwargs
        d[String(k)] = v
    end
    return d
end
numarray(description; kwargs...) = prop("array", description; items = Dict{String,Any}("type" => "number"), kwargs...)

schema(properties; required = String[]) = Dict{String,Any}(
    "type" => "object", "properties" => properties, "required" => required, "additionalProperties" => false)

const READ_ONLY = Dict{String,Any}("readOnlyHint" => true, "destructiveHint" => false,
                                   "idempotentHint" => true, "openWorldHint" => false)

const GEOMETRY_PROPERTIES = Dict{String,Any}(
    "preset" => prop("string", "Built-in rotor to start from: apc_10x5 (default, small propeller), nasa_hover_rotor (helicopter), nrel_5mw (wind turbine), or custom (requires the three geometry arrays). Any other field overrides the preset.";
                     enum = ["apc_10x5", "nasa_hover_rotor", "nrel_5mw", "custom"]),
    "rotor_type" => prop("string", "Analysis convention: propeller (efficiency, CT, CQ, CP, J), helicopter (figure of merit, CT, CP; use for hover with Vinf_m_s = 0), or windturbine (CP, CT vs tip speed ratio, wind turbine sign conventions). Defaults to the preset's type, or propeller for custom geometry.";
                         enum = ["propeller", "helicopter", "windturbine"]),
    "Rtip_m" => prop("number", "Tip radius in meters."; exclusiveMinimum = 0),
    "Rhub_over_Rtip" => prop("number", "Hub radius as a fraction of tip radius."; exclusiveMinimum = 0, exclusiveMaximum = 1),
    "num_blades" => prop("integer", "Number of blades."; minimum = 1),
    "precone_deg" => prop("number", "Precone angle in degrees (positive tilts blades downstream)."),
    "r_over_R" => numarray("Normalized radial stations r/R, strictly increasing, all greater than Rhub_over_Rtip and at most 1.0. Give r_over_R, chord_over_R and twist_deg together."; minItems = 2),
    "chord_over_R" => numarray("Chord at each station as a fraction of tip radius. Same length as r_over_R."; minItems = 2),
    "twist_deg" => numarray("Blade twist (geometric pitch angle from the rotor plane) at each station in degrees. Same length as r_over_R."; minItems = 2),
    "airfoil" => prop("string", "One polar file applied at every station (name from list_airfoils, or an absolute path). Default: the preset's airfoils, or naca4412.dat for custom geometry."),
    "airfoils" => prop("array", "Polar file per station (same length as r_over_R) for rotors with several airfoils along the span."; items = Dict{String,Any}("type" => "string")),
    "tip_correction" => prop("string", "Prandtl tip/hub loss model. Default prandtl_tip_hub."; enum = ["prandtl_tip_hub", "prandtl_tip", "none"]),
    "mach_correction" => prop("boolean", "Apply a Prandtl-Glauert compressibility correction to the airfoil coefficients. Default false."),
    "rotation_correction" => prop("boolean", "Apply the Du-Selig / Eggers 3D rotational stall-delay correction. Default false."),
    "airfoil_shape" => prop("string", "NACA 4-digit profile used only to draw the 3D blade surface (export_blade_vtk), e.g. 4412. CCBlade itself only uses polars."),
)

const OPERATING_PROPERTIES = Dict{String,Any}(
    "Vinf_m_s" => prop("number", "Axial freestream or wind speed in m/s. Use 0 for hover (with rotor_type helicopter). Presets have a default."; minimum = 0),
    "rpm" => prop("number", "Rotational speed in revolutions per minute."; exclusiveMinimum = 0),
    "tip_speed_ratio" => prop("number", "Alternative to rpm (wind turbines): Omega R / Vinf."; exclusiveMinimum = 0),
    "advance_ratio" => prop("number", "Alternative to Vinf_m_s (propellers): J = V / (n D) with n in rev/s; needs rpm."; minimum = 0),
    "pitch_deg" => prop("number", "Collective pitch added to the twist at every station, degrees. Default 0."),
    "rho_kg_m3" => prop("number", "Air density in kg/m^3. Default 1.225."; exclusiveMinimum = 0),
    "speed_of_sound_m_s" => prop("number", "Speed of sound in m/s, used for tip Mach and the Mach correction. Default 340.3."; exclusiveMinimum = 0),
    "dynamic_viscosity_Pa_s" => prop("number", "Dynamic viscosity in Pa*s, used for Reynolds number. Default 1.81e-5."; exclusiveMinimum = 0),
    "yaw_deg" => prop("number", "Wind turbine only: yaw misalignment in degrees. Default 0."),
    "tilt_deg" => prop("number", "Wind turbine only: shaft tilt in degrees. Default 0 (5 for nrel_5mw)."),
    "hub_height_m" => prop("number", "Wind turbine only: hub height in meters for wind shear. Default 2 Rtip (90 for nrel_5mw)."; exclusiveMinimum = 0),
    "shear_exponent" => prop("number", "Wind turbine only: power-law wind shear exponent. Default 0 (0.2 for nrel_5mw)."; minimum = 0),
    "n_azimuth" => prop("integer", "Wind turbine only: number of azimuth positions averaged when shear, tilt or yaw are present. Default 4."; minimum = 1, maximum = 72),
)

const SWEEP_PROPERTIES = Dict{String,Any}(
    "sweep_variable" => prop("string", "Operating variable to sweep. rpm and Vinf_m_s work for every rotor type; advance_ratio is for propellers (needs rpm); tip_speed_ratio is for wind turbines (needs Vinf_m_s); pitch_deg sweeps collective.";
                             enum = collect(SWEEP_VARIABLES)),
    "values" => numarray("Sweep points in the units of sweep_variable (dimensionless for advance_ratio and tip_speed_ratio)."; minItems = 2, maxItems = 200),
)

merged(dicts...) = merge(Dict{String,Any}(), dicts...)

# ----------------------------------------------------------------------------
# Implementations
# ----------------------------------------------------------------------------

function list_presets_impl(_)
    return Dict{String,Any}(
        "presets" => preset_summaries(),
        "usage" => "Pass preset=<name> to any analysis tool; add other fields to override parts of it, or pass r_over_R, chord_over_R and twist_deg for a custom rotor. Operating-point fields not given fall back to the preset's default operating point.",
    )
end

function list_airfoils_impl(_)
    return Dict{String,Any}(
        "airfoil_dir" => AIRFOIL_DIR,
        "airfoils" => list_airfoils(),
        "format" => "Header line, Reynolds number line, Mach number line, then columns alpha, cl, cd covering -180 to 180 degrees; alpha may be in degrees or radians (detected automatically).",
    )
end

function analyze_rotor_impl(p)
    g = resolve_geometry(p)
    op = resolve_operating(p, g)
    metrics, out = evaluate(g, op)
    result = Dict{String,Any}("geometry" => geometry_info(g), "performance" => metrics)
    getbool(p, "include_sections", false) && (result["sections"] = section_table(g, out))
    return result
end

function sweep_result(g, rows, variable)
    key = best_metric_key(g.rotor_type)
    result = Dict{String,Any}("geometry" => geometry_info(g), "sweep_variable" => variable, "rows" => rows)
    i = best_point(rows, key)
    i === nothing || (result["best_" * key] = Dict{String,Any}(key => rows[i][key], variable => rows[i][variable],
                                                              "thrust_N" => rows[i]["thrust_N"], "power_W" => rows[i]["power_W"]))
    return result
end

function sweep_rotor_impl(p)
    g = resolve_geometry(p)
    variable = getstr(p, "sweep_variable", "")
    values = getvec(p, "values")
    values === nothing && fail("values is required")
    rows = sweep(g, p, variable, values)
    return sweep_result(g, rows, variable)
end

convert_units_impl(p) = convert_units(p)

function plot_geometry_impl(p)
    g = resolve_geometry(p)
    return saved_plot(Dict{String,Any}("geometry" => geometry_info(g), "stations" => geometry_arrays(g)),
                      plot_geometry_png(g), g.preset, "geometry")
end

function plot_performance_impl(p)
    g = resolve_geometry(p)
    variable = getstr(p, "sweep_variable", "")
    values = getvec(p, "values")
    values === nothing && fail("values is required")
    rows = sweep(g, p, variable, values)
    return saved_plot(sweep_result(g, rows, variable), plot_performance_png(rows, variable, g.rotor_type, g),
                      g.preset, "performance_vs_$(variable)")
end

function plot_spanwise_impl(p)
    g = resolve_geometry(p)
    op = resolve_operating(p, g)
    metrics, out = evaluate(g, op)
    return saved_plot(Dict{String,Any}("geometry" => geometry_info(g), "performance" => metrics),
                      plot_spanwise_png(g, out, op), g.preset, "spanwise")
end

function plot_airfoil_impl(p)
    name = getstr(p, "airfoil", "")
    isempty(name) && fail("airfoil is required")
    rng = getvec(p, "alpha_range_deg")
    rng = rng === nothing ? (-30.0, 30.0) : (length(rng) == 2 ? (rng[1], rng[2]) : fail("alpha_range_deg must be [min, max]"))
    af = load_airfoil(name)
    summary = Dict{String,Any}("airfoil" => name, "description" => String(strip(af.info)), "Re" => af.Re, "Mach" => af.Mach,
                               "alpha_range_plotted_deg" => collect(rng),
                               "max_cl" => sig(maximum(af.cl), 4), "min_cd" => sig(minimum(af.cd), 4),
                               "max_lift_to_drag" => sig(maximum(af.cl ./ max.(af.cd, 1e-6)), 4))
    return saved_plot(summary, plot_airfoil_png(name; alpha_range_deg = rng), splitext(basename(name))[1], "airfoil")
end

function optimize_rotor_impl(p)
    g = resolve_geometry(p)
    op = resolve_operating(p, g)
    result, g1, op1, out0, out1 = optimize_rotor(g, op, p)
    result["initial_geometry"] = geometry_info(g)
    getbool(p, "return_plot", true) || return result
    return saved_plot(result, plot_optimization_png(g, g1, out0, out1), g.preset, "optimization")
end

function export_blade_vtk_impl(p)
    g = resolve_geometry(p)
    out = nothing
    op_note = "no operating point given, so no loads attached"
    if getbool(p, "attach_loads", true)
        try
            op = resolve_operating(p, g)
            _, out = evaluate(g, op)
            op_note = "loads attached for Vinf=$(sig(op.Vinf, 4)) m/s, $(sig(op.rpm, 5)) rpm" *
                      (isempty(op.defaults_used) ? "" : " (preset defaults used for: $(join(op.defaults_used, ", ")))")
        catch e
            e isa ToolError || rethrow()
        end
    end
    dir = getstr(p, "output_dir", OUTPUT_DIR)
    name = getstr(p, "name", replace(g.preset, r"[^A-Za-z0-9_-]" => "_") * "_blade")
    result = export_blade_vtk(g; out = out, shape = getstr(p, "airfoil_shape", g.airfoil_shape), output_dir = dir, name = name)
    result["geometry"] = geometry_info(g)
    result["loads"] = op_note
    return result
end

# ----------------------------------------------------------------------------
# Tool declarations
# ----------------------------------------------------------------------------

const CCBLADE_TOOLS = MCPTool[
    MCPTool(
        name = "list_presets", title = "List built-in rotors",
        description = "List the built-in rotor presets (APC 10x5 propeller, NASA hover rotor, NREL 5 MW wind turbine) with their geometry summary and default operating point.",
        input_schema = schema(Dict{String,Any}()), handler = guarded(list_presets_impl), annotations = READ_ONLY),
    MCPTool(
        name = "list_airfoils", title = "List airfoil polars",
        description = "List the airfoil lift/drag polar files bundled with CCBlade, with their Reynolds number, angle range and which presets use them.",
        input_schema = schema(Dict{String,Any}()), handler = guarded(list_airfoils_impl), annotations = READ_ONLY),
    MCPTool(
        name = "analyze_rotor", title = "Analyze a rotor at one operating point",
        description = "Blade element momentum (BEM) analysis with CCBlade.jl at a single operating point for a propeller, hovering rotor or wind turbine. Returns thrust, torque, power, the convention's coefficients (efficiency/CT/CQ/CP, figure of merit, or CP/CT vs tip speed ratio) and optionally the spanwise distribution. With no arguments it analyzes the APC 10x5 propeller at its default operating point.",
        input_schema = schema(merged(GEOMETRY_PROPERTIES, OPERATING_PROPERTIES, Dict{String,Any}(
            "include_sections" => prop("boolean", "Also return per-station angle of attack, inflow angle, cl, cd, loads and induction factors. Default false.")))),
        handler = guarded(analyze_rotor_impl), annotations = READ_ONLY),
    MCPTool(
        name = "sweep_rotor", title = "Sweep an operating variable",
        description = "Evaluate the rotor over a list of rpm, freestream speed, advance ratio, tip speed ratio or collective pitch values. Returns one row of performance metrics per point plus the best point (max efficiency, figure of merit or CP). Use plot_performance for the same sweep as a picture.",
        input_schema = schema(merged(GEOMETRY_PROPERTIES, OPERATING_PROPERTIES, SWEEP_PROPERTIES); required = ["sweep_variable", "values"]),
        handler = guarded(sweep_rotor_impl), annotations = READ_ONLY),
    MCPTool(
        name = "convert_rotor_units", title = "Convert rotor quantities",
        description = "Convert between dimensional and nondimensional rotor quantities using CCBlade's conventions: thrust/CT, torque/CQ, power/CP, rpm/omega/rev-per-second/tip speed, freestream/advance ratio/tip speed ratio, plus efficiency, figure of merit and tip Mach. Give whatever you know; everything derivable is returned with the formulas used.",
        input_schema = schema(Dict{String,Any}(
            "convention" => prop("string", "Normalization convention. Default propeller."; enum = collect(CONVENTIONS)),
            "Rtip_m" => prop("number", "Tip radius in meters (default 0.127, the APC 10x5)."; exclusiveMinimum = 0),
            "diameter_m" => prop("number", "Rotor diameter in meters, alternative to Rtip_m."; exclusiveMinimum = 0),
            "precone_deg" => prop("number", "Precone in degrees. Default 0."),
            "rho_kg_m3" => prop("number", "Air density. Default 1.225."; exclusiveMinimum = 0),
            "speed_of_sound_m_s" => prop("number", "Speed of sound for tip Mach. Default 340.3."; exclusiveMinimum = 0),
            "rpm" => prop("number", "Rotation speed in rpm (give at most one rotation-speed input)."; exclusiveMinimum = 0),
            "omega_rad_s" => prop("number", "Rotation speed in rad/s."; exclusiveMinimum = 0),
            "n_rev_s" => prop("number", "Rotation speed in revolutions per second."; exclusiveMinimum = 0),
            "tip_speed_m_s" => prop("number", "Blade tip speed in m/s."; exclusiveMinimum = 0),
            "Vinf_m_s" => prop("number", "Freestream or wind speed in m/s (give at most one freestream input)."; minimum = 0),
            "advance_ratio" => prop("number", "Propeller advance ratio J = V/(nD); needs a rotation speed."; minimum = 0),
            "tip_speed_ratio" => prop("number", "Wind turbine tip speed ratio; needs a rotation speed."; exclusiveMinimum = 0),
            "thrust_N" => prop("number", "Thrust in newtons."), "CT" => prop("number", "Thrust coefficient."),
            "torque_Nm" => prop("number", "Torque in N*m."), "CQ" => prop("number", "Torque coefficient."),
            "power_W" => prop("number", "Power in watts."), "CP" => prop("number", "Power coefficient."))),
        handler = guarded(convert_units_impl), annotations = READ_ONLY),
    MCPTool(
        name = "plot_geometry", title = "Plot blade geometry",
        description = "Image of the chord and twist distributions and the blade planform, plus the station arrays as JSON. Accepts the same geometry inputs as analyze_rotor. The PNG is also saved to the mcp/output folder and its path returned as image_file.",
        input_schema = schema(GEOMETRY_PROPERTIES), handler = guarded(plot_geometry_impl), annotations = READ_ONLY),
    MCPTool(
        name = "plot_performance", title = "Plot performance curves",
        description = "Sweep one operating variable (as in sweep_rotor) and return an image of efficiency or figure of merit or CP, the force coefficients, and thrust and power versus that variable, together with the numeric rows. The PNG is also saved to the mcp/output folder and its path returned as image_file.",
        input_schema = schema(merged(GEOMETRY_PROPERTIES, OPERATING_PROPERTIES, SWEEP_PROPERTIES); required = ["sweep_variable", "values"]),
        handler = guarded(plot_performance_impl), annotations = READ_ONLY),
    MCPTool(
        name = "plot_spanwise", title = "Plot spanwise distributions",
        description = "Image of the spanwise normal and tangential loads, angle of attack and inflow angle, cl and cd, and induction factors at one operating point, plus the integrated performance. The PNG is also saved to the mcp/output folder and its path returned as image_file.",
        input_schema = schema(merged(GEOMETRY_PROPERTIES, OPERATING_PROPERTIES)), handler = guarded(plot_spanwise_impl), annotations = READ_ONLY),
    MCPTool(
        name = "plot_airfoil", title = "Plot an airfoil polar",
        description = "Image of cl and cd versus angle of attack and the drag polar for one bundled airfoil file (see list_airfoils). The PNG is also saved to the mcp/output folder and its path returned as image_file.",
        input_schema = schema(Dict{String,Any}(
            "airfoil" => prop("string", "Polar file name from list_airfoils, or an absolute path."),
            "alpha_range_deg" => numarray("[min, max] angle of attack window in degrees to plot. Default [-30, 30]; the files cover -180 to 180."; minItems = 2, maxItems = 2));
            required = ["airfoil"]),
        handler = guarded(plot_airfoil_impl), annotations = READ_ONLY),
    MCPTool(
        name = "optimize_rotor", title = "Optimize chord, twist and rpm",
        description = "Gradient-based blade optimization with SNOW.jl (Ipopt, exact derivatives by ForwardDiff through CCBlade). Design variables are chord/R and twist at a few control points along the span (smoothly interpolated) and optionally rpm. Objectives: min_power or max_efficiency subject to thrust >= thrust_min_N, max_thrust subject to power <= power_max_W, or max_figure_of_merit (hover) subject to thrust >= thrust_min_N. Returns initial and optimized performance, the optimized geometry arrays (which can be passed back to any other tool) and a comparison image. The PNG is also saved to the mcp/output folder and its path returned as image_file. Typical run time is a few seconds.",
        input_schema = schema(merged(GEOMETRY_PROPERTIES, OPERATING_PROPERTIES, Dict{String,Any}(
            "objective" => prop("string", "What to optimize. Default min_power."; enum = collect(OBJECTIVES)),
            "thrust_min_N" => prop("number", "Minimum thrust constraint in newtons (required for min_power, max_efficiency and max_figure_of_merit)."; exclusiveMinimum = 0),
            "power_max_W" => prop("number", "Maximum shaft power constraint in watts (required for max_thrust, optional otherwise)."; exclusiveMinimum = 0),
            "design_variables" => prop("string", "Which distributions to vary. Default chord_and_twist."; enum = collect(DESIGN_VARIABLE_SETS)),
            "optimize_rpm" => prop("boolean", "Also treat rpm as a design variable within rpm_bounds. Default false."),
            "n_control_points" => prop("integer", "Number of spanwise control points per distribution. Default 5."; minimum = 2, maximum = 12),
            "chord_over_R_bounds" => numarray("[min, max] bounds on chord/R. Default [0.02, 0.35]."; minItems = 2, maxItems = 2),
            "twist_deg_bounds" => numarray("[min, max] bounds on twist in degrees. Default [-5, 60]."; minItems = 2, maxItems = 2),
            "rpm_bounds" => numarray("[min, max] rpm bounds when optimize_rpm is true. Default 0.5x to 1.5x the starting rpm."; minItems = 2, maxItems = 2),
            "max_iter" => prop("integer", "Ipopt iteration limit. Default 200."; minimum = 1, maximum = 2000),
            "tol" => prop("number", "Ipopt convergence tolerance on the scaled problem. Default 1e-4."; exclusiveMinimum = 0),
            "ipopt_options" => prop("object", "Extra Ipopt options passed through verbatim, e.g. {\"limited_memory_max_history\": 30}."),
            "return_plot" => prop("boolean", "Include the before/after comparison image. Default true.")))),
        handler = guarded(optimize_rotor_impl),
        annotations = Dict{String,Any}("readOnlyHint" => true, "destructiveHint" => false, "idempotentHint" => true, "openWorldHint" => false)),
    MCPTool(
        name = "export_blade_vtk", title = "Export the 3D blade for ParaView",
        description = "Write the rotor as a multiblock VTK file (all blades lofted from a NACA 4-digit section shape, plus a hub) with chord, twist and, when an operating point is available, spanwise loads and angle of attack attached as point data. Returns the file path to open in ParaView.",
        input_schema = schema(merged(GEOMETRY_PROPERTIES, OPERATING_PROPERTIES, Dict{String,Any}(
            "attach_loads" => prop("boolean", "Run the analysis at the given (or preset default) operating point and attach loads as point data. Default true."),
            "output_dir" => prop("string", "Directory to write into. Default: the mcp/output folder next to the server."),
            "name" => prop("string", "Base file name without extension. Default <preset>_blade.")))),
        handler = guarded(export_blade_vtk_impl),
        annotations = Dict{String,Any}("readOnlyHint" => false, "destructiveHint" => false, "idempotentHint" => true, "openWorldHint" => false)),
]
