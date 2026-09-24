# Gradient-based blade optimization with SNOW (Ipopt) and ForwardDiff through CCBlade.
#
# Design variables are chord/R and twist at a few control points along the span,
# interpolated to the analysis stations with an Akima spline, optionally plus rpm.

const OBJECTIVES = ("min_power", "max_efficiency", "max_thrust", "max_figure_of_merit")
const DESIGN_VARIABLE_SETS = ("chord_and_twist", "chord", "twist")

"""
    optimize_rotor(geom, op, params) -> (result::Dict, geom_opt, op_opt, out_initial, out_optimized)

Solve, for the given operating point,

    minimize   objective(chord, twist[, rpm])
    subject to thrust >= thrust_min_N      (if given)
               power  <= power_max_W       (if given)
               bounds on chord/R, twist and rpm
"""
function optimize_rotor(g::Geometry, op::OperatingPoint, p)
    op.rotor_type == "windturbine" && fail("optimize_rotor supports rotor_type propeller and helicopter (wind turbine optimization is not wired up yet)")

    objective = getstr(p, "objective", "min_power")
    objective in OBJECTIVES || fail("objective must be one of $(join(OBJECTIVES, ", "))")
    Tmin = getnum(p, "thrust_min_N", NaN)
    Pmax = getnum(p, "power_max_W", NaN)
    if objective in ("min_power", "max_efficiency", "max_figure_of_merit") && isnan(Tmin)
        fail("$objective needs thrust_min_N; without a thrust requirement the optimum is a rotor that does nothing")
    end
    objective == "max_thrust" && isnan(Pmax) && fail("max_thrust needs power_max_W")
    objective == "max_efficiency" && op.Vinf <= 0 && fail("max_efficiency needs Vinf_m_s > 0; for hover use max_figure_of_merit or min_power")
    objective == "max_figure_of_merit" && op.rotor_type != "helicopter" && fail("max_figure_of_merit needs rotor_type helicopter")
    !isnan(Tmin) && Tmin <= 0 && fail("thrust_min_N must be positive")
    !isnan(Pmax) && Pmax <= 0 && fail("power_max_W must be positive")

    dvs = getstr(p, "design_variables", "chord_and_twist")
    dvs in DESIGN_VARIABLE_SETS || fail("design_variables must be one of $(join(DESIGN_VARIABLE_SETS, ", "))")
    use_c = dvs != "twist"; use_t = dvs != "chord"
    opt_rpm = getbool(p, "optimize_rpm", false)
    nctrl = getint(p, "n_control_points", 5)
    2 <= nctrl <= 12 || fail("n_control_points must be between 2 and 12")
    cb = something(getvec(p, "chord_over_R_bounds"), [0.02, 0.35])
    tb = something(getvec(p, "twist_deg_bounds"), [-5.0, 60.0])
    rb = something(getvec(p, "rpm_bounds"), [0.5, 1.5] .* op.rpm)
    for (nm, b) in (("chord_over_R_bounds", cb), ("twist_deg_bounds", tb), ("rpm_bounds", rb))
        (length(b) == 2 && b[1] < b[2]) || fail("$nm must be [min, max] with min < max")
    end
    cb[1] > 0 || fail("chord_over_R_bounds must be positive")
    rb[1] > 0 || fail("rpm_bounds must be positive")
    max_iter = getint(p, "max_iter", 200)
    1 <= max_iter <= 2000 || fail("max_iter must be between 1 and 2000")
    tol = getnum(p, "tol", 1e-4)
    extra_ipopt = present(p, "ipopt_options") ? Dict{String,Any}(String(k) => v for (k, v) in pairs(p["ipopt_options"])) : Dict{String,Any}()

    rR, r, Rtip, rotor, afs = g.r_over_R, g.r, g.Rtip, g.rotor, g.airfoils
    ctrl = collect(range(rR[1], rR[end], length = nctrl))
    c0 = clamp.(akima(rR, g.chord_over_R, ctrl), cb[1], cb[2])
    t0 = clamp.(akima(rR, g.twist_deg, ctrl), tb[1], tb[2]) .* (pi / 180)

    x0 = Float64[]; lx = Float64[]; ux = Float64[]
    use_c && (append!(x0, c0); append!(lx, fill(cb[1], nctrl)); append!(ux, fill(cb[2], nctrl)))
    use_t && (append!(x0, t0); append!(lx, fill(tb[1] * pi / 180, nctrl)); append!(ux, fill(tb[2] * pi / 180, nctrl)))
    opt_rpm && (push!(x0, 1.0); push!(lx, rb[1] / op.rpm); push!(ux, rb[2] / op.rpm))   # rpm scaled by the initial rpm

    function unpack(x)
        i = 0
        cR = g.chord_over_R; tw = g.twist_deg .* (pi / 180); rpm = op.rpm
        if use_c
            cR = akima(ctrl, x[i+1:i+nctrl], rR); i += nctrl
        end
        if use_t
            tw = akima(ctrl, x[i+1:i+nctrl], rR); i += nctrl
        end
        opt_rpm && (rpm = x[i+1] * op.rpm)
        return cR, tw, rpm
    end

    ops0 = ccblade_ops(g, op)
    precone = g.precone_deg * pi / 180
    Rp = Rtip * cos(precone)
    A = pi * Rp^2
    out_initial = solve.(Ref(rotor), g.sections, ops0)
    T0, Q0 = thrusttorque(rotor, g.sections, out_initial)
    P0 = abs(Q0 * op.Omega) + 1e-9
    Tref = isnan(Tmin) ? abs(T0) + 1e-9 : Tmin

    neval = Ref(0)
    function func!(gcon, x)
        neval[] += 1
        cR, tw, rpm = unpack(x)
        Omega = rpm * pi / 30
        sections = Section.(r, cR .* Rtip, tw, afs)
        ops = opt_rpm ? ccblade_ops(g, op, Omega) : ops0
        out = solve.(Ref(rotor), sections, ops)
        T, Q = thrusttorque(rotor, sections, out)
        P = Q * Omega
        f = if objective == "min_power"
            P / P0
        elseif objective == "max_efficiency"
            -(T * op.Vinf) / P
        elseif objective == "max_thrust"
            -T / Tref
        else
            Vt = Omega * Rp
            CT = T / (op.rho * A * Vt^2); CP = P / (op.rho * A * Vt^3)
            -(CT > 0 ? CT^1.5 / (sqrt(2) * CP) : CT)   # negative thrust: push thrust up instead of NaN
        end
        k = 0
        isnan(Tmin) || (k += 1; gcon[k] = 1 - T / Tmin)
        isnan(Pmax) || (k += 1; gcon[k] = P / Pmax - 1)
        return f
    end

    ng = (!isnan(Tmin)) + (!isnan(Pmax))
    mkpath(OUTPUT_DIR)
    # SNOW uses Ipopt's limited-memory BFGS Hessian. With the default history of 6 the
    # dual infeasibility stalls around 1e-1 on this problem; with a history of 20 (more
    # than the number of design variables, so effectively a full quasi-Newton update) it
    # converges in a few dozen iterations. The objective is scaled to O(1), so the
    # "acceptable" criteria below are a safety net rather than the usual exit.
    ipopt = Dict{String,Any}(
        "max_iter" => max_iter, "tol" => tol,
        "limited_memory_max_history" => 20,
        # "Acceptable" exit: objective stagnant to 1e-6 (relative, since it is scaled to
        # O(1)) for 5 iterations with constraints met to 1e-4. Near bound-constrained
        # optima the quasi-Newton dual infeasibility can hover at 1e-2 while the design
        # has stopped changing; without this Ipopt would run to max_iter for nothing.
        "acceptable_tol" => 1e-1, "acceptable_obj_change_tol" => 1e-6, "acceptable_iter" => 5,
        "acceptable_constr_viol_tol" => 1e-4,
        "mu_strategy" => "adaptive",
        "print_level" => 0, "sb" => "yes", "output_file" => joinpath(OUTPUT_DIR, "ipopt.out"),
    )
    for (k, v) in extra_ipopt
        ipopt[k] = v isa Integer ? Int(v) : v isa AbstractFloat ? Float64(v) : String(v)
    end
    options = SNOW.Options(solver = SNOW.IPOPT(ipopt), derivatives = SNOW.ForwardAD())
    t = @elapsed begin
        xopt, fopt, status, _ = SNOW.minimize(func!, x0, ng, lx, ux, fill(-Inf, ng), zeros(ng), options)
    end

    cR1, tw1, rpm1 = unpack(xopt)
    g1 = rebuild_geometry(g; chord_over_R = cR1, twist_deg = tw1 .* (180 / pi))
    op1 = with_rpm(op, g, rpm1)
    m0, out0 = evaluate(g, op)
    m1, out1 = evaluate(g1, op1)

    satisfied = Dict{String,Any}()
    if !isnan(Tmin)
        satisfied["thrust_N"] = m1["thrust_N"]; satisfied["thrust_min_N"] = Tmin
        satisfied["thrust_constraint_met"] = m1["thrust_N"] !== nothing && m1["thrust_N"] >= Tmin * (1 - 1e-3)
    end
    if !isnan(Pmax)
        satisfied["power_W"] = m1["power_W"]; satisfied["power_max_W"] = Pmax
        satisfied["power_constraint_met"] = m1["power_W"] !== nothing && m1["power_W"] <= Pmax * (1 + 1e-3)
    end

    result = Dict{String,Any}(
        "status" => String(status),
        "converged" => status in (:Solve_Succeeded, :Solved_To_Acceptable_Level),
        "objective" => objective,
        "objective_value_scaled" => sig(fopt),
        "design_variables" => dvs,
        "optimize_rpm" => opt_rpm,
        "n_control_points" => nctrl,
        "control_point_r_over_R" => [sig(v, 4) for v in ctrl],
        "function_evaluations" => neval[],
        "wall_time_s" => sig(t, 3),
        "solver" => "Ipopt via SNOW.jl, ForwardDiff derivatives, limited-memory Hessian",
        "constraints" => satisfied,
        "initial" => m0,
        "optimized" => m1,
        "optimized_geometry" => geometry_arrays(g1),
    )
    opt_rpm && (result["optimized_geometry"]["rpm"] = sig(rpm1))
    if !result["converged"]
        result["hint"] = "Ipopt stopped with $(status). Try more max_iter, looser bounds, or check that thrust_min_N / power_max_W are achievable for this rotor size and rpm."
    end
    return result, g1, op1, out0, out1
end
