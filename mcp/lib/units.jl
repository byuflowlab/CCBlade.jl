# Conversions between dimensional loads and rotor coefficients, and between the
# rotation-speed and freestream measures, using CCBlade's nondimensionalization.

const CONVENTIONS = ("propeller", "helicopter", "windturbine")
const ROTATION_INPUTS = ("rpm", "omega_rad_s", "n_rev_s", "tip_speed_m_s")
const VELOCITY_INPUTS = ("Vinf_m_s", "advance_ratio", "tip_speed_ratio")

const CONVENTION_FORMULAS = Dict(
    "propeller" => ["n = rpm/60, D = 2 R cos(precone)", "CT = T / (rho n^2 D^4)", "CQ = Q / (rho n^2 D^5)",
                    "CP = P / (rho n^3 D^5) = 2 pi CQ", "J = V / (n D)", "efficiency = T V / P = J CT / CP"],
    "helicopter" => ["A = pi (R cos(precone))^2, Vtip = Omega R cos(precone)", "CT = T / (rho A Vtip^2)",
                     "CP = P / (rho A Vtip^3) = CQ", "CQ = Q / (rho A Vtip^2 R)", "FM = CT^1.5 / (sqrt(2) CP)"],
    "windturbine" => ["A = pi (R cos(precone))^2, q = 0.5 rho V^2", "CP = P / (q A V)", "CT = T / (q A)",
                      "CQ = Q / (q A R)", "tip speed ratio = Omega R / V"],
)

"""
    convert_units(params) -> Dict

Fill in every quantity derivable from the given ones. Rotation speed may be given as
one of `rpm`, `omega_rad_s`, `n_rev_s`, `tip_speed_m_s`; freestream as one of
`Vinf_m_s`, `advance_ratio`, `tip_speed_ratio`; loads as `thrust_N` or `CT`,
`torque_Nm` or `CQ`, `power_W` or `CP`.
"""
function convert_units(p)
    conv = getstr(p, "convention", "propeller")
    conv in CONVENTIONS || fail("convention must be one of $(join(CONVENTIONS, ", "))")
    assumptions = String[]

    Rtip = if present(p, "diameter_m")
        getnum(p, "diameter_m", 0.0) / 2
    elseif present(p, "Rtip_m")
        getnum(p, "Rtip_m", 0.0)
    else
        push!(assumptions, "Rtip_m defaulted to $(APC_RTIP) m (APC 10x5)")
        APC_RTIP
    end
    Rtip > 0 || fail("Rtip_m / diameter_m must be positive")
    precone = getnum(p, "precone_deg", 0.0)
    rho = getnum(p, "rho_kg_m3", 1.225)
    present(p, "rho_kg_m3") || push!(assumptions, "rho_kg_m3 defaulted to 1.225")
    asound = getnum(p, "speed_of_sound_m_s", 340.3)
    Rp = Rtip * cos(precone * pi / 180)
    D = 2Rp
    A = pi * Rp^2

    out = Dict{String,Any}("convention" => conv, "Rtip_m" => sig(Rtip), "diameter_m" => sig(2Rtip),
                           "rotor_plane_radius_m" => sig(Rp), "disk_area_m2" => sig(A), "rho_kg_m3" => rho)

    # rotation speed
    rot = [k for k in ROTATION_INPUTS if present(p, k)]
    length(rot) <= 1 || fail("Give only one of $(join(ROTATION_INPUTS, ", "))")
    Omega = nothing
    if !isempty(rot)
        k = rot[1]; v = getnum(p, k, 0.0)
        v > 0 || fail("$k must be positive")
        Omega = k == "rpm" ? v * pi / 30 : k == "omega_rad_s" ? v : k == "n_rev_s" ? 2pi * v : v / Rp
        out["rpm"] = sig(Omega * 30 / pi); out["omega_rad_s"] = sig(Omega); out["n_rev_s"] = sig(Omega / 2pi)
        out["tip_speed_m_s"] = sig(Omega * Rp); out["tip_mach"] = sig(Omega * Rtip / asound, 3)
    end

    # freestream, and the two ratios that link freestream and rotation speed
    V = present(p, "Vinf_m_s") ? getnum(p, "Vinf_m_s", 0.0) : nothing
    V !== nothing && V < 0 && fail("Vinf_m_s must be non-negative")
    links = [k for k in ("advance_ratio", "tip_speed_ratio") if present(p, k)]
    length(links) <= 1 || fail("Give only one of advance_ratio and tip_speed_ratio")
    if !isempty(links)
        k = links[1]; v = getnum(p, k, 0.0)
        v > 0 || fail("$k must be positive")
        if Omega !== nothing && V !== nothing
            push!(assumptions, "$k was ignored because both a rotation speed and Vinf_m_s were given; it is recomputed from them")
        elseif Omega !== nothing
            V = k == "advance_ratio" ? v * (Omega / 2pi) * D : Omega * Rp / v
        elseif V !== nothing
            V > 0 || fail("$k with Vinf_m_s = 0 does not determine a rotation speed")
            Omega = k == "advance_ratio" ? 2pi * V / (v * D) : v * V / Rp
            out["rpm"] = sig(Omega * 30 / pi); out["omega_rad_s"] = sig(Omega); out["n_rev_s"] = sig(Omega / 2pi)
            out["tip_speed_m_s"] = sig(Omega * Rp); out["tip_mach"] = sig(Omega * Rtip / asound, 3)
        else
            fail("$k needs either a rotation speed (rpm, omega_rad_s, n_rev_s, tip_speed_m_s) or Vinf_m_s to convert")
        end
    end
    if V !== nothing
        out["Vinf_m_s"] = sig(V)
        if Omega !== nothing
            out["advance_ratio"] = sig(V / ((Omega / 2pi) * D))
            V > 0 && (out["tip_speed_ratio"] = sig(Omega * Rp / V))
        end
    end

    # normalizers: T = CT*kT, Q = CQ*kQ, P = CP*kP
    ks = if conv == "propeller"
        Omega === nothing ? nothing : (n = Omega / 2pi; (rho * n^2 * D^4, rho * n^2 * D^5, rho * n^3 * D^5))
    elseif conv == "helicopter"
        Omega === nothing ? nothing : (Vt = Omega * Rp; (rho * A * Vt^2, rho * A * Vt^2 * Rp, rho * A * Vt^3))
    else
        (V === nothing || V == 0) ? nothing : (q = 0.5 * rho * V^2; (q * A, q * A * Rp, q * A * V))
    end
    need = conv == "windturbine" ? "a wind speed (Vinf_m_s)" : "a rotation speed (rpm, omega_rad_s, n_rev_s or tip_speed_m_s)"

    T = present(p, "thrust_N") ? getnum(p, "thrust_N", 0.0) :
        present(p, "CT") ? (ks === nothing && fail("Converting CT needs $need"); getnum(p, "CT", 0.0) * ks[1]) : nothing

    Q = nothing; P = nothing
    if present(p, "torque_Nm")
        Q = getnum(p, "torque_Nm", 0.0)
        present(p, "power_W") && push!(assumptions, "both torque_Nm and power_W given; torque used, power recomputed as Q*Omega")
    elseif present(p, "power_W")
        P = getnum(p, "power_W", 0.0)
    elseif present(p, "CQ")
        ks === nothing && fail("Converting CQ needs $need")
        Q = getnum(p, "CQ", 0.0) * ks[2]
    elseif present(p, "CP")
        ks === nothing && fail("Converting CP needs $need")
        P = getnum(p, "CP", 0.0) * ks[3]
    end
    if Omega !== nothing
        Q !== nothing && P === nothing && (P = Q * Omega)
        P !== nothing && Q === nothing && (Q = P / Omega)
    elseif (Q !== nothing) != (P !== nothing)
        push!(assumptions, "torque and power cannot be related without a rotation speed")
    end

    T !== nothing && (out["thrust_N"] = sig(T))
    Q !== nothing && (out["torque_Nm"] = sig(Q))
    P !== nothing && (out["power_W"] = sig(P))
    if ks !== nothing
        T !== nothing && (out["CT"] = sig(T / ks[1]))
        Q !== nothing && (out["CQ"] = sig(Q / ks[2]))
        P !== nothing && (out["CP"] = sig(P / ks[3]))
    end

    # derived figures of merit
    if conv == "propeller" && T !== nothing && P !== nothing && V !== nothing && P > 0
        out["efficiency"] = sig(T * V / P)
    end
    if conv == "helicopter" && T !== nothing && P !== nothing && ks !== nothing
        CT = T / ks[1]; CP = P / ks[3]
        CT > 0 && CP > 0 && (out["figure_of_merit"] = sig(CT^1.5 / (sqrt(2) * CP)))
        out["disk_loading_N_m2"] = sig(T / A)
        P > 0 && (out["power_loading_N_per_W"] = sig(T / P))
    end

    out["formulas"] = CONVENTION_FORMULAS[conv]
    isempty(assumptions) || (out["assumptions"] = assumptions)
    return out
end
