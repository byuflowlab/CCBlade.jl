# Exercises RotorTools directly (no MCP), prints key numbers and writes PNG/VTK files to
# mcp/output. Run from the repository root:
#
#   julia --startup-file=no --project=mcp mcp/test/run_lib_tests.jl
#
# The first run compiles Plots and SNOW and takes a few minutes; later runs take seconds.

include(joinpath(@__DIR__, "..", "lib", "RotorTools.jl"))
using .RotorTools
using Test

mkpath(OUTPUT_DIR)
save(name, bytes) = (path = joinpath(OUTPUT_DIR, name); write(path, bytes); println("  wrote $path ($(length(bytes)) bytes)"))

@testset "RotorTools" begin
    @testset "presets and airfoils" begin
        @test length(preset_summaries()) == 3
        afs = list_airfoils()
        @test any(a -> a["file"] == "naca4412.dat" && a["angle_units_in_file"] == "radians", afs)
        @test any(a -> a["file"] == "DU21_A17.dat" && a["angle_units_in_file"] == "degrees", afs)
        println("  airfoils: ", join([a["file"] for a in afs], ", "))
    end

    @testset "APC 10x5 propeller (tutorial values)" begin
        g = resolve_geometry(Dict{String,Any}())
        op = resolve_operating(Dict{String,Any}("Vinf_m_s" => 5.0, "rpm" => 5400.0), g)
        m, out = evaluate(g, op)
        println("  APC @ 5 m/s, 5400 rpm: ", (m["thrust_N"], m["torque_Nm"], m["efficiency"]))
        @test isapprox(m["thrust_N"], 3.13; rtol = 0.02)
        @test 0.4 < m["efficiency"] < 0.55
        @test length(section_table(g, out)) == 18
        rows = sweep(g, Dict{String,Any}("rpm" => 5400.0), "advance_ratio", [0.1, 0.3, 0.5, 0.6, 0.7])
        i = best_point(rows, "efficiency")
        println("  best efficiency ", rows[i]["efficiency"], " at J = ", rows[i]["advance_ratio"])
        @test 0.45 <= rows[i]["advance_ratio"] <= 0.65
        @test 0.6 < rows[i]["efficiency"] < 0.72
    end

    @testset "NREL 5 MW wind turbine (how-to values)" begin
        g = resolve_geometry(Dict{String,Any}("preset" => "nrel_5mw"))
        @test g.rotor_type == "windturbine"
        op = resolve_operating(Dict{String,Any}(), g)
        @test "tip_speed_ratio" in op.defaults_used
        m, out = evaluate(g, op)
        println("  NREL @ 10 m/s, TSR 7.55: CP=", m["CP"], " CT=", m["CT"], " P=", m["power_kW"], " kW")
        @test 0.40 < m["CP"] < 0.55
        @test 3000 < m["power_kW"] < 6500
        rows = sweep(g, Dict{String,Any}(), "tip_speed_ratio", [4.0, 6.0, 8.0, 10.0])
        @test length(rows) == 4 && all(r -> r["CP"] !== nothing, rows)
    end

    @testset "hover rotor" begin
        g = resolve_geometry(Dict{String,Any}("preset" => "nasa_hover_rotor"))
        op = resolve_operating(Dict{String,Any}("pitch_deg" => 8.0), g)
        m, _ = evaluate(g, op)
        println("  hover rotor @ 8 deg collective: T=", m["thrust_N"], " N, FM=", m["figure_of_merit"])
        @test m["thrust_N"] > 0
        @test 0.3 < m["figure_of_merit"] < 0.9
        rows = sweep(g, Dict{String,Any}(), "pitch_deg", [2.0, 6.0, 10.0, 14.0])
        @test issorted([r["thrust_N"] for r in rows])
    end

    @testset "custom geometry and validation" begin
        p = Dict{String,Any}("r_over_R" => [0.2, 0.5, 1.0], "chord_over_R" => [0.12, 0.10, 0.05],
                             "twist_deg" => [30.0, 18.0, 9.0], "Rtip_m" => 0.3, "num_blades" => 4, "rpm" => 3000, "Vinf_m_s" => 0.0,
                             "rotor_type" => "helicopter", "mach_correction" => true, "tip_correction" => "prandtl_tip")
        g = resolve_geometry(p)
        @test g.preset == "custom" && g.num_blades == 4
        m, _ = evaluate(g, resolve_operating(p, g))
        @test m["thrust_N"] > 0
        @test_throws ToolError resolve_geometry(Dict{String,Any}("r_over_R" => [0.5, 1.0], "chord_over_R" => [0.1]))
        @test_throws ToolError resolve_geometry(Dict{String,Any}("r_over_R" => [0.05, 1.0], "chord_over_R" => [0.1, 0.1], "twist_deg" => [1.0, 1.0]))
        @test_throws ToolError resolve_operating(Dict{String,Any}("Vinf_m_s" => 5.0), resolve_geometry(Dict{String,Any}("preset" => "custom", "r_over_R" => [0.5, 1.0], "chord_over_R" => [0.1, 0.1], "twist_deg" => [1.0, 1.0])))
        @test_throws ToolError load_airfoil("nope.dat")
    end

    @testset "unit conversion" begin
        u = convert_units(Dict{String,Any}("convention" => "propeller", "Rtip_m" => 0.127, "rpm" => 5400, "Vinf_m_s" => 5.0, "thrust_N" => 3.13288, "torque_Nm" => 0.0585127))
        println("  units: CT=", u["CT"], " CP=", u["CP"], " J=", u["advance_ratio"], " eta=", u["efficiency"])
        @test isapprox(u["CT"], 0.0758555; rtol = 1e-3)
        @test isapprox(u["CP"], 0.0350461; rtol = 1e-3)
        @test isapprox(u["efficiency"], 0.473414; rtol = 1e-3)
        back = convert_units(Dict{String,Any}("convention" => "propeller", "Rtip_m" => 0.127, "rpm" => 5400, "CT" => u["CT"]))
        @test isapprox(back["thrust_N"], 3.13288; rtol = 1e-3)
        h = convert_units(Dict{String,Any}("convention" => "helicopter", "Rtip_m" => 0.656, "tip_speed_m_s" => 55.0, "thrust_N" => 40.0, "power_W" => 900.0))
        @test haskey(h, "figure_of_merit") && haskey(h, "rpm")
        w = convert_units(Dict{String,Any}("convention" => "windturbine", "Rtip_m" => 63.0, "Vinf_m_s" => 10.0, "tip_speed_ratio" => 7.55, "CP" => 0.48))
        @test isapprox(w["power_W"], 0.48 * 0.5 * 1.225 * 1000 * pi * 63^2; rtol = 1e-6)
        @test_throws ToolError convert_units(Dict{String,Any}("CT" => 0.1))
    end

    @testset "plots" begin
        g = resolve_geometry(Dict{String,Any}())
        save("test_geometry.png", plot_geometry_png(g))
        rows = sweep(g, Dict{String,Any}("rpm" => 5400.0), "advance_ratio", collect(0.1:0.1:0.8))
        save("test_performance.png", plot_performance_png(rows, "advance_ratio", "propeller", g))
        op = resolve_operating(Dict{String,Any}("Vinf_m_s" => 5.0, "rpm" => 5400.0), g)
        _, out = evaluate(g, op)
        save("test_spanwise.png", plot_spanwise_png(g, out, op))
        save("test_airfoil.png", plot_airfoil_png("naca4412.dat"))
        gt = resolve_geometry(Dict{String,Any}("preset" => "nrel_5mw"))
        rows = sweep(gt, Dict{String,Any}(), "tip_speed_ratio", collect(3.0:1.0:12.0))
        save("test_performance_turbine.png", plot_performance_png(rows, "tip_speed_ratio", "windturbine", gt))
        @test true
    end

    @testset "optimization" begin
        g = resolve_geometry(Dict{String,Any}())
        op = resolve_operating(Dict{String,Any}("Vinf_m_s" => 10.0, "rpm" => 5400.0), g)
        t = @elapsed (res, g1, op1, out0, out1) = optimize_rotor(g, op, Dict{String,Any}("objective" => "min_power", "thrust_min_N" => 2.0))
        println("  min_power: ", res["status"], " in ", res["function_evaluations"], " evals, ", round(t; digits = 1), " s; power ",
                res["initial"]["power_W"], " -> ", res["optimized"]["power_W"], " W at thrust ", res["optimized"]["thrust_N"], " N")
        @test res["converged"]
        @test res["optimized"]["thrust_N"] >= 2.0 * 0.999
        @test res["optimized"]["power_W"] < res["initial"]["power_W"] || res["initial"]["thrust_N"] < 2.0
        save("test_optimization.png", plot_optimization_png(g, g1, out0, out1))

        res2, = optimize_rotor(g, op, Dict{String,Any}("objective" => "max_efficiency", "thrust_min_N" => 2.0, "design_variables" => "twist", "optimize_rpm" => true))
        println("  max_efficiency (twist + rpm): ", res2["status"], " eta ", res2["initial"]["efficiency"], " -> ", res2["optimized"]["efficiency"], " rpm ", res2["optimized_geometry"]["rpm"])
        @test res2["converged"]

        gh = resolve_geometry(Dict{String,Any}("preset" => "nasa_hover_rotor"))
        oph = resolve_operating(Dict{String,Any}("pitch_deg" => 8.0), gh)
        res3, = optimize_rotor(gh, oph, Dict{String,Any}("objective" => "max_figure_of_merit", "thrust_min_N" => 30.0))
        println("  max FM (hover): ", res3["status"], " FM ", res3["initial"]["figure_of_merit"], " -> ", res3["optimized"]["figure_of_merit"])
        @test res3["converged"]
        @test_throws ToolError optimize_rotor(g, op, Dict{String,Any}("objective" => "max_efficiency"))
    end

    @testset "vtk export" begin
        g = resolve_geometry(Dict{String,Any}())
        op = resolve_operating(Dict{String,Any}("Vinf_m_s" => 5.0, "rpm" => 5400.0), g)
        _, out = evaluate(g, op)
        r = export_blade_vtk(g; out = out, name = "test_apc_blade")
        println("  vtk: ", r["main_file"])
        @test isfile(r["main_file"])
        gt = resolve_geometry(Dict{String,Any}("preset" => "nrel_5mw"))
        r2 = export_blade_vtk(gt; name = "test_nrel_blade")
        @test isfile(r2["main_file"])
    end
end
println("LIB_TESTS_DONE")
