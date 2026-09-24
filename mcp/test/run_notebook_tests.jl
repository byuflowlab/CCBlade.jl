# Headless check of the logic inside mcp/notebook/rotor_explorer.jl.
#
# Pluto cells cannot be executed outside a Pluto session, so this file repeats the
# computational body of each cell with fixed "slider" values. It catches the errors that
# actually break the notebook -- wrong keys, wrong argument order, a preset whose default
# operating point lacks a field -- without needing a browser.
#
#   julia --project=mcp mcp/test/run_notebook_tests.jl

using Test
using JSON3

include(joinpath(@__DIR__, "..", "lib", "RotorTools.jl"))
using .RotorTools

# Mirrors the notebook's geometry cell.
function notebook_geometry(preset; num_blades = nothing, Rtip = nothing, chord_scale = 1.0,
                           twist_scale = 1.0, twist_offset = 0.0, tip_correction = "prandtl_tip_hub",
                           mach_correction = false, rotation_correction = false)
    base = RotorTools.PRESETS[preset]
    return RotorTools.resolve_geometry(Dict{String,Any}(
        "preset" => preset,
        "Rtip_m" => Rtip === nothing ? base.Rtip : Rtip,
        "num_blades" => num_blades === nothing ? base.num_blades : num_blades,
        "r_over_R" => base.r_over_R,
        "chord_over_R" => base.chord_over_R .* chord_scale,
        "twist_deg" => base.twist_deg .* twist_scale .+ twist_offset,
        "airfoils" => base.airfoils,
        "rotor_type" => base.rotor_type,
        "tip_correction" => tip_correction,
        "mach_correction" => mach_correction,
        "rotation_correction" => rotation_correction,
    ))
end

# Mirrors the notebook's operating-point cell.
function notebook_op_params(preset; Vinf, rpm, pitch, rho = 1.225)
    d = RotorTools.PRESETS[preset].default_op
    p = Dict{String,Any}("Vinf_m_s" => Vinf, "rpm" => rpm, "pitch_deg" => pitch, "rho_kg_m3" => rho)
    for k in ("yaw_deg", "tilt_deg", "hub_height_m", "shear_exponent", "n_azimuth")
        haskey(d, k) && (p[k] = d[k])
    end
    return p
end

# Mirrors the slider-range cell, which must work for a preset that has no "rpm" default.
function notebook_defaults(preset)
    base = RotorTools.PRESETS[preset]
    d = base.default_op
    rpm_default = if haskey(d, "rpm")
        Float64(d["rpm"])
    else
        V = Float64(d["Vinf_m_s"])
        Float64(d["tip_speed_ratio"]) * V / base.Rtip * 30 / pi
    end
    return (rpm = rpm_default, Vinf = Float64(d["Vinf_m_s"]), pitch = Float64(get(d, "pitch_deg", 0.0)))
end

@testset "notebook cells" begin

    @testset "slider defaults for $preset" for preset in ("apc_10x5", "nasa_hover_rotor", "nrel_5mw")
        d = notebook_defaults(preset)
        @test d.rpm > 0
        @test d.Vinf >= 0
        @test isfinite(d.pitch)
    end

    @testset "geometry + evaluate + plots for $preset" for preset in ("apc_10x5", "nasa_hover_rotor", "nrel_5mw")
        d = notebook_defaults(preset)
        g = notebook_geometry(preset)
        p = notebook_op_params(preset; Vinf = d.Vinf, rpm = d.rpm, pitch = d.pitch)
        op = RotorTools.resolve_operating(p, g)
        metrics, out = RotorTools.evaluate(g, op)
        @test metrics["thrust_N"] !== nothing

        # The metrics table cell indexes these keys by rotor type; all must be present.
        keys_shown = if g.rotor_type == "propeller"
            ("thrust_N", "torque_Nm", "power_W", "efficiency", "CT", "CP", "advance_ratio", "tip_mach")
        elseif g.rotor_type == "helicopter"
            ("thrust_N", "torque_Nm", "power_W", "figure_of_merit", "CT", "CP", "disk_loading_N_m2", "tip_mach")
        else
            ("thrust_N", "torque_Nm", "power_kW", "CP", "CT", "CQ", "tip_speed_ratio", "tip_mach")
        end
        @test all(haskey(metrics, k) for k in keys_shown)

        @test length(RotorTools.plot_geometry_png(g)) > 1000
        @test length(RotorTools.plot_spanwise_png(g, out, op)) > 1000
    end

    @testset "geometry sliders change the rotor" begin
        g0 = notebook_geometry("apc_10x5")
        g1 = notebook_geometry("apc_10x5"; num_blades = 4, Rtip = 0.2, chord_scale = 1.5,
                               twist_scale = 0.8, twist_offset = 2.0)
        @test g1.num_blades == 4
        @test g1.Rtip ≈ 0.2
        @test g1.chord_over_R ≈ g0.chord_over_R .* 1.5
        @test g1.twist_deg ≈ g0.twist_deg .* 0.8 .+ 2.0
    end

    @testset "correction toggles" begin
        for tc in ("prandtl_tip_hub", "prandtl_tip", "none")
            g = notebook_geometry("apc_10x5"; tip_correction = tc)
            @test g.corrections["tip_correction"] == tc
        end
        g = notebook_geometry("apc_10x5"; mach_correction = true, rotation_correction = true)
        @test g.corrections["mach_correction"]
        @test g.corrections["rotation_correction"]
    end

    @testset "sweep cell" begin
        g = notebook_geometry("apc_10x5")
        p = notebook_op_params("apc_10x5"; Vinf = 5.0, rpm = 5400.0, pitch = 0.0)
        rows = RotorTools.sweep(g, p, "advance_ratio", collect(range(0.1, 0.7, length = 5)))
        @test length(rows) == 5
        key = RotorTools.best_metric_key(g.rotor_type)
        @test key == "efficiency"
        best = RotorTools.best_point(rows, key)
        @test best !== nothing
        @test rows[best][key] > 0
        @test length(RotorTools.plot_performance_png(rows, "advance_ratio", g.rotor_type, g)) > 1000
    end

    @testset "sweep cell for the turbine preset" begin
        g = notebook_geometry("nrel_5mw")
        p = notebook_op_params("nrel_5mw"; Vinf = 10.0, rpm = notebook_defaults("nrel_5mw").rpm, pitch = 0.0)
        rows = RotorTools.sweep(g, p, "tip_speed_ratio", [5.0, 7.0, 9.0])
        @test length(rows) == 3
        @test RotorTools.best_metric_key(g.rotor_type) == "CP"
        @test length(RotorTools.plot_performance_png(rows, "tip_speed_ratio", g.rotor_type, g)) > 1000
    end

    @testset "optimization cell" begin
        g = notebook_geometry("apc_10x5")
        p = notebook_op_params("apc_10x5"; Vinf = 10.0, rpm = 5400.0, pitch = 0.0)
        op = RotorTools.resolve_operating(p, g)
        res, g1, op1, out0, out1 = RotorTools.optimize_rotor(g, op, Dict{String,Any}(
            "objective" => "min_power", "design_variables" => "chord_and_twist",
            "optimize_rpm" => false, "thrust_min_N" => 2.0))
        # Keys the result cell reads.
        @test haskey(res, "status") && haskey(res, "objective") && haskey(res, "function_evaluations")
        @test res["converged"]
        @test length(RotorTools.plot_optimization_png(g, g1, out0, out1)) > 1000
    end

    @testset "VTK cell" begin
        g = notebook_geometry("apc_10x5")
        p = notebook_op_params("apc_10x5"; Vinf = 5.0, rpm = 5400.0, pitch = 0.0)
        op = RotorTools.resolve_operating(p, g)
        _, out = RotorTools.evaluate(g, op)
        info = RotorTools.export_blade_vtk(g; out = out, name = "notebook_test")
        # Keys the result cell reads.
        @test haskey(info, "main_file") && haskey(info, "num_blades") && haskey(info, "point_data")
        @test isfile(info["main_file"])
        @test info["num_blades"] == g.num_blades
    end

    @testset "design JSON handoff cell" begin
        g = notebook_geometry("apc_10x5"; num_blades = 3, chord_scale = 1.2)
        p = notebook_op_params("apc_10x5"; Vinf = 6.0, rpm = 5000.0, pitch = 1.0)
        op = RotorTools.resolve_operating(p, g)
        d = RotorTools.geometry_arrays(g)
        merge!(d, Dict{String,Any}("Vinf_m_s" => RotorTools.sig(op.Vinf, 5),
                                   "rpm" => RotorTools.sig(op.rpm, 5),
                                   "pitch_deg" => RotorTools.sig(op.pitch_deg, 5),
                                   "rho_kg_m3" => RotorTools.sig(op.rho, 5)))
        json = sprint(io -> JSON3.pretty(io, d))
        @test occursin("chord_over_R", json)

        # The whole point of the handoff: the JSON must be valid input for the MCP tools.
        # JSON3.read gives Symbol keys; the tools take strings, as an MCP client sends them.
        back = Dict{String,Any}(String(k) => v for (k, v) in pairs(JSON3.read(json)))
        g2 = RotorTools.resolve_geometry(back)
        @test g2.num_blades == 3
        @test g2.chord_over_R ≈ g.chord_over_R rtol = 1e-4
        op2 = RotorTools.resolve_operating(back, g2)
        @test op2.rpm ≈ 5000.0
        @test op2.pitch_deg ≈ 1.0
    end
end
