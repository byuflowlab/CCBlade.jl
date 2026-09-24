# Plots rendered to PNG bytes (returned to MCP clients as image content).

const XLABELS = Dict("rpm" => "rpm", "Vinf_m_s" => "Vinf (m/s)", "advance_ratio" => "advance ratio J = V/(nD)",
                     "tip_speed_ratio" => "tip speed ratio", "pitch_deg" => "collective pitch (deg)")

function png_bytes(plt)
    io = IOBuffer()
    show(io, MIME("image/png"), plt)
    return take!(io)
end

nan_or(x) = x === nothing ? NaN : Float64(x)
column(rows, key) = [nan_or(get(r, key, nothing)) for r in rows]

function geometry_title(g::Geometry)
    names = unique(g.airfoil_names)
    afs = length(names) <= 2 ? join(names, ", ") : "$(length(names)) airfoils ($(names[end]) at the tip)"
    return "$(g.preset): B=$(g.num_blades), R=$(sig(g.Rtip, 4)) m, $(afs)"
end

"Chord and twist distributions plus a planform view (pitch axis at quarter chord)."
function plot_geometry_png(g::Geometry; title = geometry_title(g))
    rR, cR, tw = g.r_over_R, g.chord_over_R, g.twist_deg
    p1 = plot(rR, cR; xlabel = "r/R", ylabel = "chord / R", marker = :circle, ms = 3, legend = false)
    p2 = plot(rR, tw; xlabel = "r/R", ylabel = "twist (deg)", marker = :circle, ms = 3, legend = false)
    le = 0.25 .* cR; te = -0.75 .* cR
    p3 = plot(rR, le; fillrange = te, fillalpha = 0.35, color = :steelblue, xlabel = "r/R",
              ylabel = "chordwise / R", legend = false, aspect_ratio = :equal)
    plot!(p3, [g.Rhub / g.Rtip, 1.0], [0.0, 0.0]; color = :black, ls = :dash)
    plot!(p3, rR, te; color = :steelblue)
    plt = plot(p1, p2, p3; layout = (1, 3), size = (1250, 400), plot_title = title, plot_titlefontsize = 11,
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)
    return png_bytes(plt)
end

"Performance curves from sweep rows: coefficients and dimensional loads versus the swept variable."
function plot_performance_png(rows, variable, rotor_type, g::Geometry)
    x = column(rows, variable)
    xl = get(XLABELS, variable, variable)
    T = column(rows, "thrust_N"); P = column(rows, "power_W")
    if rotor_type == "propeller"
        p1 = plot(x, column(rows, "efficiency"); xlabel = xl, ylabel = "efficiency", legend = false, marker = :circle, ms = 3)
        p2 = plot(x, [column(rows, "CT") column(rows, "CP")]; xlabel = xl, label = ["CT" "CP"], marker = :circle, ms = 3)
        p3 = plot(x, T; xlabel = xl, ylabel = "thrust (N)", label = "thrust", marker = :circle, ms = 3, color = 1)
        plot!(twinx(p3), x, P; ylabel = "power (W)", label = "power", color = 2, marker = :circle, ms = 3, legend = :topleft)
    elseif rotor_type == "helicopter"
        p1 = plot(x, column(rows, "figure_of_merit"); xlabel = xl, ylabel = "figure of merit", legend = false, marker = :circle, ms = 3)
        p2 = plot(x, [column(rows, "CT") column(rows, "CP")]; xlabel = xl, label = ["CT" "CP"], marker = :circle, ms = 3)
        p3 = plot(x, T; xlabel = xl, ylabel = "thrust (N)", label = "thrust", marker = :circle, ms = 3, color = 1)
        plot!(twinx(p3), x, P; ylabel = "power (W)", label = "power", color = 2, marker = :circle, ms = 3, legend = :topleft)
    else
        p1 = plot(x, column(rows, "CP"); xlabel = xl, ylabel = "CP", legend = false, marker = :circle, ms = 3)
        p2 = plot(x, column(rows, "CT"); xlabel = xl, ylabel = "CT", legend = false, marker = :circle, ms = 3)
        p3 = plot(x, P ./ 1e3; xlabel = xl, ylabel = "power (kW)", label = "power", marker = :circle, ms = 3, color = 1)
        plot!(twinx(p3), x, T ./ 1e3; ylabel = "thrust (kN)", label = "thrust", color = 2, marker = :circle, ms = 3, legend = :topleft)
    end
    plt = plot(p1, p2, p3; layout = (1, 3), size = (1250, 400), plot_title = geometry_title(g), plot_titlefontsize = 11,
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, right_margin = 12Plots.mm)
    return png_bytes(plt)
end

function operating_title(op::OperatingPoint)
    s = "Vinf=$(sig(op.Vinf, 4)) m/s, $(sig(op.rpm, 5)) rpm"
    op.pitch_deg != 0 && (s *= ", pitch=$(sig(op.pitch_deg, 4)) deg")
    return s
end

"Spanwise distributions at one operating point: loads, angles, coefficients, inductions."
function plot_spanwise_png(g::Geometry, out, op::OperatingPoint)
    o = out isa AbstractMatrix ? out[:, 1] : out
    x = g.r_over_R
    p1 = plot(x, [field(o, :Np) field(o, :Tp)]; xlabel = "r/R", ylabel = "load (N/m)", label = ["normal Np" "tangential Tp"], marker = :circle, ms = 3)
    p2 = plot(x, [field(o, :alpha) .* (180 / pi) field(o, :phi) .* (180 / pi)]; xlabel = "r/R", ylabel = "angle (deg)", label = ["angle of attack" "inflow angle"], marker = :circle, ms = 3)
    p3 = plot(x, field(o, :cl); xlabel = "r/R", ylabel = "cl", label = "cl", marker = :circle, ms = 3, color = 1)
    plot!(twinx(p3), x, field(o, :cd); ylabel = "cd", label = "cd", color = 2, marker = :circle, ms = 3, legend = :topleft)
    p4 = plot(x, [field(o, :a) field(o, :ap)]; xlabel = "r/R", ylabel = "induction factor", label = ["axial a" "tangential a'"], marker = :circle, ms = 3)
    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 750), plot_title = geometry_title(g) * "  |  " * operating_title(op),
               plot_titlefontsize = 11, left_margin = 6Plots.mm, bottom_margin = 4Plots.mm, right_margin = 10Plots.mm)
    return png_bytes(plt)
end

"Lift and drag polar of one airfoil file over a chosen angle-of-attack window."
function plot_airfoil_png(name; alpha_range_deg = (-30.0, 30.0))
    af = load_airfoil(name)
    a = af.alpha .* (180 / pi)
    lo, hi = alpha_range_deg
    lo < hi || fail("alpha_range_deg must be [min, max] with min < max")
    m = (a .>= lo) .& (a .<= hi)
    any(m) || fail("No polar points inside alpha_range_deg; the file covers $(sig(minimum(a),4)) to $(sig(maximum(a),4)) deg")
    p1 = plot(a[m], af.cl[m]; xlabel = "alpha (deg)", ylabel = "cl", legend = false, marker = :circle, ms = 2)
    p2 = plot(a[m], af.cd[m]; xlabel = "alpha (deg)", ylabel = "cd", legend = false, marker = :circle, ms = 2)
    p3 = plot(af.cd[m], af.cl[m]; xlabel = "cd", ylabel = "cl", legend = false, marker = :circle, ms = 2)
    plt = plot(p1, p2, p3; layout = (1, 3), size = (1250, 400), plot_title = "$(name): $(strip(af.info)) (Re=$(af.Re), Mach=$(af.Mach))",
               plot_titlefontsize = 11, left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)
    return png_bytes(plt)
end

"Initial versus optimized chord, twist, loads and angle of attack."
function plot_optimization_png(g0::Geometry, g1::Geometry, out0, out1, label0 = "initial", label1 = "optimized")
    x = g0.r_over_R
    o0 = out0 isa AbstractMatrix ? out0[:, 1] : out0
    o1 = out1 isa AbstractMatrix ? out1[:, 1] : out1
    lab = [label0 label1]
    p1 = plot(x, [g0.chord_over_R g1.chord_over_R]; xlabel = "r/R", ylabel = "chord / R", label = lab, marker = :circle, ms = 3)
    p2 = plot(x, [g0.twist_deg g1.twist_deg]; xlabel = "r/R", ylabel = "twist (deg)", label = lab, marker = :circle, ms = 3)
    p3 = plot(x, [field(o0, :Np) field(o1, :Np)]; xlabel = "r/R", ylabel = "normal load Np (N/m)", label = lab, marker = :circle, ms = 3)
    p4 = plot(x, [field(o0, :alpha) field(o1, :alpha)] .* (180 / pi); xlabel = "r/R", ylabel = "angle of attack (deg)", label = lab, marker = :circle, ms = 3)
    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 750), plot_title = "Optimization: " * geometry_title(g0),
               plot_titlefontsize = 11, left_margin = 6Plots.mm, bottom_margin = 4Plots.mm)
    return png_bytes(plt)
end
