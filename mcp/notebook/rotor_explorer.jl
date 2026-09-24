### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 1a000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PlutoUI
    using JSON3
    include(joinpath(@__DIR__, "..", "lib", "RotorTools.jl"))
    using .RotorTools
    using .RotorTools: Plots

    "Render an exception as a Markdown admonition (md\"\" interpolation is unreliable here)."
    error_box(title, err) = Markdown.parse(
        "!!! warning \"" * title * "\"\n    " *
        replace(sprint(showerror, err), "\n" => "\n    "))

    md"Environment ready."
end

# ╔═╡ 1a000000-0000-0000-0000-000000000002
md"""
# CCBlade Rotor Explorer

Move the sliders; every plot below re-solves the blade element momentum equations and
redraws. This notebook is a thin front end over `mcp/lib/RotorTools.jl` — the exact
module the MCP server calls, so a slider here and a question to Claude run the same code.

$(PlutoUI.TableOfContents(title = "Contents", depth = 2))
"""

# ╔═╡ 1a000000-0000-0000-0000-000000000003
md"## 1. Rotor and operating point"

# ╔═╡ 1a000000-0000-0000-0000-000000000004
@bind preset Select(
    ["apc_10x5" => "APC Thin Electric 10x5 (propeller)",
     "nasa_hover_rotor" => "NASA hover rotor (helicopter)",
     "nrel_5mw" => "NREL 5 MW (wind turbine)"],
    default = "apc_10x5")

# ╔═╡ 1a000000-0000-0000-0000-000000000005
# Ranges depend on the preset, so the sliders are rebuilt whenever the preset changes.
begin
    base = RotorTools.PRESETS[preset]
    op_default = base.default_op
    rpm_default = if haskey(op_default, "rpm")
        Float64(op_default["rpm"])
    else
        # A turbine preset gives tip speed ratio instead; convert it for the slider.
        V = Float64(op_default["Vinf_m_s"])
        Float64(op_default["tip_speed_ratio"]) * V / base.Rtip * 30 / pi
    end
    vinf_default = Float64(op_default["Vinf_m_s"])
    pitch_default = Float64(get(op_default, "pitch_deg", 0.0))
    rpm_range = range(0.05 * rpm_default, 2.0 * rpm_default, length = 120)
    vinf_max = max(2.0 * vinf_default, 0.1 * rpm_default * pi / 30 * base.Rtip)
    vinf_range = range(0.0, vinf_max, length = 120)
    md"Preset **$(preset)** loaded: $(base.num_blades) blades, R = $(round(base.Rtip, digits=3)) m."
end

# ╔═╡ 1a000000-0000-0000-0000-000000000006
md"""
| | |
|---|---|
| rotation speed (rpm) | $(@bind rpm Slider(rpm_range, default = rpm_default, show_value = true)) |
| freestream Vinf (m/s) | $(@bind Vinf Slider(vinf_range, default = vinf_default, show_value = true)) |
| collective pitch (deg) | $(@bind pitch Slider(-10:0.25:25, default = pitch_default, show_value = true)) |
| air density (kg/m³) | $(@bind rho Slider(0.4:0.005:1.4, default = 1.225, show_value = true)) |
"""

# ╔═╡ 1a000000-0000-0000-0000-000000000007
md"## 2. Geometry scaling"

# ╔═╡ 1a000000-0000-0000-0000-000000000008
md"""
These scale the preset's own chord and twist distributions, so you can explore a family
of blades without typing arrays. `blades` and `radius` change the rotor outright.

| | |
|---|---|
| number of blades | $(@bind num_blades Slider(1:12, default = base.num_blades, show_value = true)) |
| tip radius (m) | $(@bind Rtip Slider(range(0.25 * base.Rtip, 2.0 * base.Rtip, length = 120), default = base.Rtip, show_value = true)) |
| chord scale | $(@bind chord_scale Slider(0.3:0.01:2.0, default = 1.0, show_value = true)) |
| twist scale | $(@bind twist_scale Slider(0.0:0.01:2.0, default = 1.0, show_value = true)) |
| twist offset (deg) | $(@bind twist_offset Slider(-10:0.25:10, default = 0.0, show_value = true)) |
"""

# ╔═╡ 1a000000-0000-0000-0000-000000000009
md"""
Corrections: tip/hub loss $(@bind tip_correction Select(["prandtl_tip_hub" => "Prandtl tip + hub", "prandtl_tip" => "Prandtl tip only", "none" => "none"]))
 · Mach $(@bind mach_correction CheckBox(default = false))
 · rotational stall delay $(@bind rotation_correction CheckBox(default = false))
"""

# ╔═╡ 1a000000-0000-0000-0000-00000000000a
# Every downstream cell depends on `geom`, so any slider above triggers a rebuild.
geom = RotorTools.resolve_geometry(Dict{String,Any}(
    "preset" => preset,
    "Rtip_m" => Rtip,
    "num_blades" => num_blades,
    "chord_over_R" => base.chord_over_R .* chord_scale,
    "twist_deg" => base.twist_deg .* twist_scale .+ twist_offset,
    "r_over_R" => base.r_over_R,
    "airfoils" => base.airfoils,
    "rotor_type" => base.rotor_type,
    "tip_correction" => tip_correction,
    "mach_correction" => mach_correction,
    "rotation_correction" => rotation_correction,
))

# ╔═╡ 1a000000-0000-0000-0000-00000000000b
op_params = Dict{String,Any}(
    "Vinf_m_s" => Vinf, "rpm" => rpm, "pitch_deg" => pitch, "rho_kg_m3" => rho,
    # Wind turbine inflow settings come from the preset default when it has them.
    [k => op_default[k] for k in ("yaw_deg", "tilt_deg", "hub_height_m", "shear_exponent", "n_azimuth")
     if haskey(op_default, k)]...,
)

# ╔═╡ 1a000000-0000-0000-0000-00000000000c
op = RotorTools.resolve_operating(op_params, geom)

# ╔═╡ 1a000000-0000-0000-0000-00000000000d
# CCBlade can fail to converge at extreme slider positions; keep the notebook alive.
result = try
    (ok = true, value = RotorTools.evaluate(geom, op))
catch err
    (ok = false, value = err)
end

# ╔═╡ 1a000000-0000-0000-0000-00000000000e
md"## 3. Performance at this operating point"

# ╔═╡ 1a000000-0000-0000-0000-00000000000f
if !result.ok
    error_box("Solver failed", result.value)
else
    metrics = result.value[1]
    keys_shown = if geom.rotor_type == "propeller"
        ("thrust_N", "torque_Nm", "power_W", "efficiency", "CT", "CP", "advance_ratio", "tip_mach")
    elseif geom.rotor_type == "helicopter"
        ("thrust_N", "torque_Nm", "power_W", "figure_of_merit", "CT", "CP", "disk_loading_N_m2", "tip_mach")
    else
        ("thrust_N", "torque_Nm", "power_kW", "CP", "CT", "CQ", "tip_speed_ratio", "tip_mach")
    end
    rows = join(["| $k | $(metrics[k]) |" for k in keys_shown if haskey(metrics, k)], "\n")
    Markdown.parse("| quantity | value |\n|---|---|\n" * rows)
end

# ╔═╡ 1a000000-0000-0000-0000-000000000010
md"## 4. Geometry"

# ╔═╡ 1a000000-0000-0000-0000-000000000011
# The library returns PNG bytes (that is what the MCP server ships to a client); Pluto
# displays them directly.
PlutoUI.Show(MIME"image/png"(), RotorTools.plot_geometry_png(geom))

# ╔═╡ 1a000000-0000-0000-0000-000000000012
md"## 5. Spanwise distributions"

# ╔═╡ 1a000000-0000-0000-0000-000000000013
result.ok ? PlutoUI.Show(MIME"image/png"(), RotorTools.plot_spanwise_png(geom, result.value[2], op)) :
            md"_(no solution at this operating point)_"

# ╔═╡ 1a000000-0000-0000-0000-000000000014
md"""
## 6. Performance sweep

The sweep holds the geometry fixed and walks one operating variable. It is the slowest
cell in the notebook, so it only runs when you press the button.
"""

# ╔═╡ 1a000000-0000-0000-0000-000000000015
md"""
sweep over $(@bind sweep_var Select(collect(RotorTools.SWEEP_VARIABLES),
    default = base.rotor_type == "windturbine" ? "tip_speed_ratio" : "advance_ratio"))
 from $(@bind sweep_lo NumberField(-50.0:0.01:1e5, default = 0.0))
 to $(@bind sweep_hi NumberField(-50.0:0.01:1e5, default = 0.8))
 in $(@bind sweep_n Slider(3:1:40, default = 15, show_value = true)) points
 $(@bind run_sweep CounterButton("Run sweep"))
"""

# ╔═╡ 1a000000-0000-0000-0000-000000000016
sweep_result = let
    run_sweep    # depend on the button so the sweep reruns when it is pressed
    if sweep_hi <= sweep_lo
        nothing
    else
        values = collect(range(sweep_lo, sweep_hi, length = sweep_n))
        try
            RotorTools.sweep(geom, op_params, sweep_var, values)
        catch err
            err
        end
    end
end

# ╔═╡ 1a000000-0000-0000-0000-000000000017
if sweep_result isa Vector
    # The caption is built outside a md"" literal on purpose: `$(sweep_var)` inside one is
    # parsed as an expression rather than as this variable.
    sweep_key = RotorTools.best_metric_key(geom.rotor_type)
    sweep_best = RotorTools.best_point(sweep_result, sweep_key)
    PlutoUI.ExperimentalLayout.vbox([
        Markdown.parse(string("Best ", sweep_key, " = **", sweep_result[sweep_best][sweep_key],
                              "** at ", sweep_var, " = **", sweep_result[sweep_best][sweep_var], "**")),
        PlutoUI.Show(MIME"image/png"(),
                     RotorTools.plot_performance_png(sweep_result, sweep_var, geom.rotor_type, geom)),
    ])
elseif sweep_result isa Exception
    error_box("Sweep failed", sweep_result)
else
    md"_Set a valid range and press **Run sweep**._"
end

# ╔═╡ 1a000000-0000-0000-0000-000000000018
md"""
## 7. Optimize (SNOW + Ipopt)

Optimization takes tens of seconds, so it is behind a button too. It varies chord and
twist at spanwise control points to improve the objective subject to a thrust floor.
Wind turbine optimization is not wired up.
"""

# ╔═╡ 1a000000-0000-0000-0000-000000000019
md"""
objective $(@bind objective Select(collect(RotorTools.OBJECTIVES), default = "min_power"))
 · design variables $(@bind design_variables Select(collect(RotorTools.DESIGN_VARIABLE_SETS), default = "chord_and_twist"))
 · also vary rpm $(@bind optimize_rpm CheckBox(default = false))

minimum thrust (N) $(@bind thrust_min NumberField(0.0:0.01:1e6, default = 2.0))
 · maximum power (W) $(@bind power_max NumberField(0.0:0.01:1e9, default = 0.0))
 · $(@bind run_opt CounterButton("Run optimization"))
"""

# ╔═╡ 1a000000-0000-0000-0000-00000000001a
opt_result = let
    run_opt
    if run_opt == 0
        nothing
    else
        p = Dict{String,Any}("objective" => objective, "design_variables" => design_variables,
                             "optimize_rpm" => optimize_rpm)
        thrust_min > 0 && (p["thrust_min_N"] = thrust_min)
        power_max > 0 && (p["power_max_W"] = power_max)
        try
            RotorTools.optimize_rotor(geom, op, p)
        catch err
            err
        end
    end
end

# ╔═╡ 1a000000-0000-0000-0000-00000000001b
if opt_result isa Tuple
    res, g_opt, op_opt, out0, out1 = opt_result
    md"""
    Status **$(res["status"])** · objective $(res["objective"]) · $(res["function_evaluations"]) evaluations

    $(PlutoUI.Show(MIME"image/png"(), RotorTools.plot_optimization_png(geom, g_opt, out0, out1)))
    """
elseif opt_result isa Exception
    error_box("Optimization failed", opt_result)
else
    md"_Press **Run optimization**._"
end

# ╔═╡ 1a000000-0000-0000-0000-00000000001c
md"""
## 8. Export the blade for ParaView

Writes a multiblock `.vtm` (plus one `.vts` per blade) under `mcp/output/`. Open the
`.vtm` in ParaView and color by `Np_N_per_m`, `alpha_deg` or `cl`.
"""

# ╔═╡ 1a000000-0000-0000-0000-00000000001d
md"""
file name $(@bind vtk_name TextField(default = "notebook_blade"))
 · geometry $(@bind vtk_source Select(["current" => "current sliders", "optimized" => "optimized (section 7)"]))
 · $(@bind run_vtk CounterButton("Write VTK"))
"""

# ╔═╡ 1a000000-0000-0000-0000-00000000001e
vtk_result = let
    run_vtk
    if run_vtk == 0
        nothing
    elseif vtk_source == "optimized" && !(opt_result isa Tuple)
        ErrorException("Run the optimization first, or choose the current geometry.")
    else
        g_out, o_out = vtk_source == "optimized" ? (opt_result[2], opt_result[5]) :
                       (geom, result.ok ? result.value[2] : nothing)
        try
            RotorTools.export_blade_vtk(g_out; out = o_out, name = vtk_name)
        catch err
            err
        end
    end
end

# ╔═╡ 1a000000-0000-0000-0000-00000000001f
if vtk_result isa Dict
    # Values are computed first: only bare-variable $(x) interpolation is reliable inside md"".
    vtk_file = basename(vtk_result["main_file"])
    vtk_blades = vtk_result["num_blades"]
    vtk_fields = join(vtk_result["point_data"], ", ")
    # Interpolation is not applied inside Markdown backtick spans, so the path is plain text.
    md"""
    Wrote **$(vtk_file)** — $(vtk_blades) blades, point data: $(vtk_fields).

    Open it from **mcp/output/** in ParaView and colour by `Np_N_per_m`, `alpha_deg` or `cl`.
    """
elseif vtk_result isa Exception
    error_box("Export failed", vtk_result)
else
    md"_Press **Write VTK**._"
end

# ╔═╡ 1a000000-0000-0000-0000-000000000020
md"""
## 9. Hand this design to the MCP server

The dictionary below is exactly what `analyze_rotor`, `optimize_rotor` or
`export_blade_vtk` accept. Copy it into a conversation with an MCP client and Claude can
pick up where the sliders left off.
"""

# ╔═╡ 1a000000-0000-0000-0000-000000000021
design_json = let
    d = RotorTools.geometry_arrays(geom)
    merge!(d, Dict{String,Any}("Vinf_m_s" => RotorTools.sig(op.Vinf, 5),
                               "rpm" => RotorTools.sig(op.rpm, 5),
                               "pitch_deg" => RotorTools.sig(op.pitch_deg, 5),
                               "rho_kg_m3" => RotorTools.sig(op.rho, 5)))
    sprint(io -> JSON3.pretty(io, d))
end

# ╔═╡ 1a000000-0000-0000-0000-000000000022
Markdown.parse("```json\n" * design_json * "\n```")

# ╔═╡ Cell order:
# ╟─1a000000-0000-0000-0000-000000000002
# ╟─1a000000-0000-0000-0000-000000000003
# ╟─1a000000-0000-0000-0000-000000000004
# ╟─1a000000-0000-0000-0000-000000000005
# ╟─1a000000-0000-0000-0000-000000000006
# ╟─1a000000-0000-0000-0000-000000000007
# ╟─1a000000-0000-0000-0000-000000000008
# ╟─1a000000-0000-0000-0000-000000000009
# ╠═1a000000-0000-0000-0000-00000000000a
# ╟─1a000000-0000-0000-0000-00000000000b
# ╟─1a000000-0000-0000-0000-00000000000c
# ╟─1a000000-0000-0000-0000-00000000000d
# ╟─1a000000-0000-0000-0000-00000000000e
# ╟─1a000000-0000-0000-0000-00000000000f
# ╟─1a000000-0000-0000-0000-000000000010
# ╟─1a000000-0000-0000-0000-000000000011
# ╟─1a000000-0000-0000-0000-000000000012
# ╟─1a000000-0000-0000-0000-000000000013
# ╟─1a000000-0000-0000-0000-000000000014
# ╟─1a000000-0000-0000-0000-000000000015
# ╟─1a000000-0000-0000-0000-000000000016
# ╟─1a000000-0000-0000-0000-000000000017
# ╟─1a000000-0000-0000-0000-000000000018
# ╟─1a000000-0000-0000-0000-000000000019
# ╟─1a000000-0000-0000-0000-00000000001a
# ╟─1a000000-0000-0000-0000-00000000001b
# ╟─1a000000-0000-0000-0000-00000000001c
# ╟─1a000000-0000-0000-0000-00000000001d
# ╟─1a000000-0000-0000-0000-00000000001e
# ╟─1a000000-0000-0000-0000-00000000001f
# ╟─1a000000-0000-0000-0000-000000000020
# ╟─1a000000-0000-0000-0000-000000000021
# ╟─1a000000-0000-0000-0000-000000000022
# ╟─1a000000-0000-0000-0000-000000000001
