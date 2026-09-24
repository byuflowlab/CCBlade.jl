# Operating points, single-point evaluation, spanwise tables and sweeps.

struct OperatingPoint
    rotor_type::String
    Vinf::Float64        # m/s
    rpm::Float64
    Omega::Float64       # rad/s
    rho::Float64         # kg/m^3
    pitch_deg::Float64   # collective pitch added to the twist
    asound::Float64      # m/s
    mu::Float64          # Pa*s
    tsr::Float64         # tip speed ratio (NaN when meaningless)
    yaw_deg::Float64
    tilt_deg::Float64
    hub_height::Float64  # m
    shear_exp::Float64
    n_azimuth::Int
    defaults_used::Vector{String}
end

const OP_KEYS = ("Vinf_m_s", "rpm", "tip_speed_ratio", "advance_ratio", "pitch_deg", "rho_kg_m3",
                 "speed_of_sound_m_s", "dynamic_viscosity_Pa_s", "yaw_deg", "tilt_deg", "hub_height_m",
                 "shear_exponent", "n_azimuth")

rotor_plane_radius(g::Geometry) = g.Rtip * cos(g.precone_deg * pi / 180)

"""
    resolve_operating(params, geom) -> OperatingPoint

Explicit inputs win; anything missing falls back to the preset's default operating
point (recorded in `defaults_used`). Rotation speed may be given as `rpm` or, for a
wind turbine, `tip_speed_ratio`; freestream as `Vinf_m_s` or, for a propeller,
`advance_ratio` (J = V/(nD), with rpm known).
"""
function resolve_operating(p, g::Geometry)
    explicit(k) = present(p, k)
    d = copy(g.default_op)
    for k in OP_KEYS
        explicit(k) && (d[k] = p[k])
    end
    used = String[k for k in OP_KEYS if haskey(d, k) && !explicit(k)]

    rho = getnum(d, "rho_kg_m3", 1.225); rho > 0 || fail("rho_kg_m3 must be positive")
    asound = getnum(d, "speed_of_sound_m_s", 340.3)
    mu = getnum(d, "dynamic_viscosity_Pa_s", 1.81e-5)
    pitch = getnum(d, "pitch_deg", 0.0)
    Rp = rotor_plane_radius(g)

    # rotation speed
    rpm = NaN; tsr = NaN
    if explicit("rpm") || (!explicit("tip_speed_ratio") && haskey(d, "rpm"))
        rpm = getnum(d, "rpm", NaN)
    elseif haskey(d, "tip_speed_ratio")
        tsr = getnum(d, "tip_speed_ratio", NaN)
    end

    # freestream
    Vinf = NaN
    if explicit("advance_ratio") && !explicit("Vinf_m_s")
        isnan(rpm) && fail("advance_ratio needs rpm")
        J = getnum(d, "advance_ratio", NaN)
        J >= 0 || fail("advance_ratio must be non-negative")
        Vinf = J * (rpm / 60) * 2 * Rp
    else
        Vinf = getnum(d, "Vinf_m_s", NaN)
    end
    isnan(Vinf) && fail("Vinf_m_s is required (freestream or wind speed in m/s; 0 for hover)")
    Vinf >= 0 || fail("Vinf_m_s must be non-negative")

    if isnan(rpm)
        isnan(tsr) && fail(g.rotor_type == "windturbine" ? "rpm or tip_speed_ratio is required" : "rpm is required")
        Vinf > 0 || fail("tip_speed_ratio needs Vinf_m_s > 0")
        tsr > 0 || fail("tip_speed_ratio must be positive")
        Omega = Vinf * tsr / Rp
        rpm = Omega * 30 / pi
    else
        rpm > 0 || fail("rpm must be positive")
        Omega = rpm * pi / 30
        tsr = Vinf > 0 ? Omega * Rp / Vinf : NaN
    end

    n_az = getint(d, "n_azimuth", 4)
    1 <= n_az <= 72 || fail("n_azimuth must be between 1 and 72")
    hub_height = getnum(d, "hub_height_m", 2.0 * g.Rtip)

    return OperatingPoint(g.rotor_type, Vinf, rpm, Omega, rho, pitch, asound, mu, tsr,
                          getnum(d, "yaw_deg", 0.0), getnum(d, "tilt_deg", 0.0), hub_height,
                          getnum(d, "shear_exponent", 0.0), n_az, used)
end

"Copy of an operating point at a different rotation speed."
function with_rpm(op::OperatingPoint, g::Geometry, rpm)
    Omega = rpm * pi / 30
    tsr = op.Vinf > 0 ? Omega * rotor_plane_radius(g) / op.Vinf : NaN
    return OperatingPoint(op.rotor_type, op.Vinf, rpm, Omega, op.rho, op.pitch_deg, op.asound, op.mu, tsr,
                          op.yaw_deg, op.tilt_deg, op.hub_height, op.shear_exp, op.n_azimuth, op.defaults_used)
end

"CCBlade operating-point objects for every station (a matrix over azimuth for turbines)."
function ccblade_ops(g::Geometry, op::OperatingPoint, Omega = op.Omega)
    pitch = op.pitch_deg * pi / 180
    precone = g.precone_deg * pi / 180
    if op.rotor_type == "windturbine"
        az = collect(range(0, 2pi, length = op.n_azimuth + 1))[1:op.n_azimuth]
        return windturbine_op.(op.Vinf, Omega, pitch, g.r, precone, op.yaw_deg * pi / 180, op.tilt_deg * pi / 180,
                               az', op.hub_height, op.shear_exp, op.rho, op.mu, op.asound)
    else
        return simple_op.(op.Vinf, Omega, g.r, op.rho; pitch = pitch, mu = op.mu, asound = op.asound, precone = precone)
    end
end

"Integrated performance metrics from thrust and torque, in the rotor type's convention."
function performance_metrics(g::Geometry, op::OperatingPoint, T, Q)
    Omega = op.Omega
    P = Q * Omega
    Rp = rotor_plane_radius(g)
    A = pi * Rp^2
    m = Dict{String,Any}(
        "thrust_N" => sig(T), "torque_Nm" => sig(Q), "power_W" => sig(P),
        "rpm" => sig(op.rpm), "Vinf_m_s" => sig(op.Vinf), "pitch_deg" => sig(op.pitch_deg),
        "rho_kg_m3" => sig(op.rho),
        "tip_speed_m_s" => sig(Omega * Rp), "tip_mach" => sig(Omega * g.Rtip / op.asound, 3),
    )
    notes = String[]
    if op.rotor_type == "propeller"
        eff, CT, CQ = nondim(T, Q, op.Vinf, Omega, op.rho, g.rotor, "propeller")
        m["advance_ratio"] = sig(op.Vinf / ((Omega / 2pi) * 2Rp))
        m["efficiency"] = sig(eff); m["CT"] = sig(CT); m["CQ"] = sig(CQ); m["CP"] = sig(2pi * CQ)
        T < 0 && push!(notes, "Negative thrust: the rotor is windmilling at this condition, so efficiency is reported as 0.")
        op.Vinf == 0 && push!(notes, "Vinf is 0 (hover): efficiency is undefined; use rotor_type helicopter for figure of merit.")
    elseif op.rotor_type == "helicopter"
        FM, CT, CP = nondim(T, Q, op.Vinf, Omega, op.rho, g.rotor, "helicopter")
        m["figure_of_merit"] = sig(FM); m["CT"] = sig(CT); m["CP"] = sig(CP)
        m["disk_loading_N_m2"] = sig(T / A); m["power_loading_N_per_W"] = sig(T / P)
        op.Vinf > 0 && push!(notes, "Helicopter normalization is meant for hover (Vinf_m_s = 0); figure of merit reported with axial inflow.")
    else
        CP, CT, CQ = nondim(T, Q, op.Vinf, Omega, op.rho, g.rotor, "windturbine")
        m["tip_speed_ratio"] = sig(op.tsr); m["CP"] = sig(CP); m["CT"] = sig(CT); m["CQ"] = sig(CQ)
        m["power_kW"] = sig(P / 1e3)
        m["azimuth_averaged_over"] = op.n_azimuth
    end
    isempty(op.defaults_used) || (m["operating_point_defaults_used"] = op.defaults_used)
    isempty(notes) || (m["notes"] = notes)
    return m
end

"""
    evaluate(geom, op) -> (metrics::Dict, outputs)

Solve the blade element momentum equations at one operating point.
"""
function evaluate(g::Geometry, op::OperatingPoint)
    ops = ccblade_ops(g, op)
    out = solve.(Ref(g.rotor), g.sections, ops)
    T, Q = thrusttorque(g.rotor, g.sections, out)
    return performance_metrics(g, op, T, Q), out
end

"Per-station table of angles, coefficients, loads and inductions (azimuth 0 for turbines)."
function section_table(g::Geometry, out)
    o = out isa AbstractMatrix ? out[:, 1] : out
    return [Dict{String,Any}(
                "r_over_R" => sig(g.r_over_R[i], 4),
                "r_m" => sig(g.r[i]),
                "chord_m" => sig(g.sections[i].chord),
                "twist_deg" => sig(g.sections[i].theta * 180 / pi),
                "alpha_deg" => sig(o[i].alpha * 180 / pi),
                "inflow_angle_deg" => sig(o[i].phi * 180 / pi),
                "cl" => sig(o[i].cl), "cd" => sig(o[i].cd),
                "normal_load_N_per_m" => sig(o[i].Np),
                "tangential_load_N_per_m" => sig(o[i].Tp),
                "axial_induction" => sig(o[i].a),
                "tangential_induction" => sig(o[i].ap),
                "local_velocity_m_s" => sig(o[i].W),
            ) for i in eachindex(g.r)]
end

# ----------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------

const SWEEP_VARIABLES = ("rpm", "Vinf_m_s", "advance_ratio", "tip_speed_ratio", "pitch_deg")
const SWEEP_CONFLICTS = Dict("rpm" => ("tip_speed_ratio",), "tip_speed_ratio" => ("rpm",),
                             "Vinf_m_s" => ("advance_ratio",), "advance_ratio" => ("Vinf_m_s",),
                             "pitch_deg" => ())

"Evaluate the rotor at each value of one operating variable; returns one metrics Dict per point."
function sweep(g::Geometry, p, variable::AbstractString, values)
    variable in SWEEP_VARIABLES || fail("sweep_variable must be one of $(join(SWEEP_VARIABLES, ", "))")
    length(values) >= 2 || fail("values must contain at least two points")
    length(values) <= 200 || fail("values is limited to 200 points per call")
    rows = Dict{String,Any}[]
    for v in values
        q = Dict{String,Any}(k => p[k] for k in OP_KEYS if present(p, k))
        for c in SWEEP_CONFLICTS[variable]
            delete!(q, c)
        end
        q[variable] = v
        op = resolve_operating(q, g)
        m, _ = evaluate(g, op)
        m[variable] = sig(v)
        push!(rows, m)
    end
    return rows
end

best_metric_key(rotor_type) = rotor_type == "propeller" ? "efficiency" :
                              rotor_type == "helicopter" ? "figure_of_merit" : "CP"

"Row index with the largest value of `key`, ignoring missing values."
function best_point(rows, key)
    idx = [i for i in eachindex(rows) if rows[i][key] !== nothing]
    isempty(idx) && return nothing
    return idx[argmax(i -> rows[i][key], idx)]
end
