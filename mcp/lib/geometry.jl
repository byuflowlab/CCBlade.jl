# Resolving a rotor geometry (Rotor + Sections) from a plain dictionary of inputs.

const ROTOR_TYPES = ("propeller", "helicopter", "windturbine")

const TIP_CORRECTIONS = Dict{String,Any}(
    "prandtl_tip_hub" => PrandtlTipHub(),
    "prandtl_tip" => PrandtlTip(),
    "none" => nothing,
)

# ----------------------------------------------------------------------------
# Airfoil polars
# ----------------------------------------------------------------------------

const AIRFOIL_CACHE = Dict{String,AlphaAF}()

function resolve_airfoil_path(name::AbstractString)
    path = isabspath(name) ? String(name) : joinpath(AIRFOIL_DIR, name)
    isfile(path) || fail("Airfoil file '$name' not found. Call list_airfoils for the bundled files, or pass an absolute path to a CCBlade-format polar file.")
    return path
end

# CCBlade polar files may store alpha in radians (naca4412.dat) or degrees (the DU
# files). Any |alpha| beyond 2*pi cannot be radians, so use that to pick.
function airfoil_file_is_degrees(path)
    raw = AlphaAF(path; radians = true)
    return maximum(abs, raw.alpha) > 2pi + 0.5
end

"Load (and cache) a polar file, converting to radians if the file is in degrees."
function load_airfoil(name::AbstractString)
    path = resolve_airfoil_path(name)
    return get!(AIRFOIL_CACHE, path) do
        AlphaAF(path; radians = !airfoil_file_is_degrees(path))
    end
end

"Describe the bundled polar files."
function list_airfoils()
    files = filter(f -> endswith(f, ".dat") || endswith(f, ".txt"), sort(readdir(AIRFOIL_DIR)))
    return map(files) do f
        path = joinpath(AIRFOIL_DIR, f)
        af = load_airfoil(f)
        Dict{String,Any}(
            "file" => f,
            "description" => String(strip(af.info)),
            "Re" => af.Re,
            "Mach" => af.Mach,
            "num_points" => length(af.alpha),
            "angle_units_in_file" => airfoil_file_is_degrees(path) ? "degrees" : "radians",
            "alpha_range_deg" => [sig(minimum(af.alpha) * 180 / pi, 4), sig(maximum(af.alpha) * 180 / pi, 4)],
            "max_cl" => sig(maximum(af.cl), 4),
            "min_cd" => sig(minimum(af.cd), 4),
            "used_by_presets" => [pr.name for pr in values(PRESETS) if f in pr.airfoils],
        )
    end
end

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

struct Geometry
    preset::String
    rotor_type::String
    rotor::Rotor
    sections::Vector           # Vector of CCBlade.Section
    r::Vector{Float64}         # m
    r_over_R::Vector{Float64}
    chord_over_R::Vector{Float64}
    twist_deg::Vector{Float64}
    Rtip::Float64
    Rhub::Float64
    num_blades::Int
    precone_deg::Float64
    airfoil_names::Vector{String}
    airfoils::Vector{AlphaAF}
    airfoil_shape::String
    corrections::Dict{String,Any}
    default_op::Dict{String,Any}
end

"""
    resolve_geometry(params) -> Geometry

Build the rotor from JSON-like inputs. Values come from, in order of precedence:
explicit entries in `params`, the chosen `preset` (default `apc_10x5`), and for a
`custom` geometry the APC scalars as fallbacks. Giving any of `r_over_R`,
`chord_over_R`, `twist_deg` requires all three.
"""
function resolve_geometry(p)
    arrays_given = any(k -> present(p, k), ("r_over_R", "chord_over_R", "twist_deg"))
    preset_name = getstr(p, "preset", arrays_given ? "custom" : "apc_10x5")
    base = if preset_name == "custom"
        arrays_given || fail("preset 'custom' needs r_over_R, chord_over_R and twist_deg")
        PRESETS["apc_10x5"]
    else
        haskey(PRESETS, preset_name) || fail("Unknown preset '$preset_name'. Call list_presets for the options, or pass geometry arrays for a custom rotor.")
        PRESETS[preset_name]
    end

    rotor_type = getstr(p, "rotor_type", preset_name == "custom" ? "propeller" : base.rotor_type)
    rotor_type in ROTOR_TYPES || fail("rotor_type must be one of $(join(ROTOR_TYPES, ", "))")

    Rtip = getnum(p, "Rtip_m", base.Rtip)
    Rtip > 0 || fail("Rtip_m must be positive (meters)")
    hub_ratio = getnum(p, "Rhub_over_Rtip", base.Rhub_over_Rtip)
    0 < hub_ratio < 1 || fail("Rhub_over_Rtip must be between 0 and 1")
    B = getint(p, "num_blades", base.num_blades)
    B >= 1 || fail("num_blades must be at least 1")
    precone_deg = getnum(p, "precone_deg", base.precone_deg)

    rR = getvec(p, "r_over_R"); cR = getvec(p, "chord_over_R"); tw = getvec(p, "twist_deg")
    if arrays_given
        (rR !== nothing && cR !== nothing && tw !== nothing) ||
            fail("Give r_over_R, chord_over_R and twist_deg together (or none of them to use a preset).")
        length(rR) == length(cR) == length(tw) ||
            fail("r_over_R, chord_over_R and twist_deg must have the same length")
        length(rR) >= 2 || fail("At least two radial stations are required")
        (issorted(rR) && allunique(rR)) || fail("r_over_R must be strictly increasing")
        (rR[1] > hub_ratio && rR[end] <= 1.0) ||
            fail("r_over_R must lie inside (Rhub_over_Rtip, 1.0]; got first=$(rR[1]), last=$(rR[end]) with Rhub_over_Rtip=$(hub_ratio)")
        all(>(0), cR) || fail("chord_over_R values must be positive")
    else
        rR, cR, tw = copy(base.r_over_R), copy(base.chord_over_R), copy(base.twist_deg)
    end
    n = length(rR)

    names = if present(p, "airfoils")
        v = getstrvec(p, "airfoils")
        length(v) == n || fail("airfoils must have one file per station ($n stations, got $(length(v)))")
        v
    elseif present(p, "airfoil")
        fill(getstr(p, "airfoil", ""), n)
    elseif length(base.airfoils) == n
        copy(base.airfoils)
    else
        fill(base.airfoils[end], n)   # custom stations on a multi-airfoil preset: use its outboard airfoil
    end

    tipname = getstr(p, "tip_correction", "prandtl_tip_hub")
    haskey(TIP_CORRECTIONS, tipname) || fail("tip_correction must be one of $(join(keys(TIP_CORRECTIONS), ", "))")
    mach = getbool(p, "mach_correction", false) ? PrandtlGlauert() : nothing
    rot = getbool(p, "rotation_correction", false) ? DuSeligEggers() : nothing

    rotor = Rotor(hub_ratio * Rtip, Rtip, B;
                  precone = precone_deg * pi / 180, turbine = rotor_type == "windturbine",
                  mach = mach, rotation = rot, tip = TIP_CORRECTIONS[tipname])
    afs = AlphaAF[load_airfoil(nm) for nm in names]
    r = rR .* Rtip
    sections = Section.(r, cR .* Rtip, tw .* (pi / 180), afs)

    corrections = Dict{String,Any}("tip_correction" => tipname,
                                   "mach_correction" => mach !== nothing,
                                   "rotation_correction" => rot !== nothing)
    default_op = preset_name == "custom" ? Dict{String,Any}() : copy(base.default_op)
    shape = getstr(p, "airfoil_shape", base.airfoil_shape)

    return Geometry(preset_name, rotor_type, rotor, sections, r, rR, cR, tw, Rtip, hub_ratio * Rtip, B,
                    precone_deg, names, afs, shape, corrections, default_op)
end

"Same rotor with new chord and/or twist distributions (used after optimization)."
function rebuild_geometry(g::Geometry; chord_over_R = g.chord_over_R, twist_deg = g.twist_deg)
    cR = collect(Float64, chord_over_R); tw = collect(Float64, twist_deg)
    sections = Section.(g.r, cR .* g.Rtip, tw .* (pi / 180), g.airfoils)
    return Geometry(g.preset == "custom" ? "custom" : g.preset * " (modified)", g.rotor_type, g.rotor, sections,
                    g.r, g.r_over_R, cR, tw, g.Rtip, g.Rhub, g.num_blades, g.precone_deg, g.airfoil_names,
                    g.airfoils, g.airfoil_shape, g.corrections, g.default_op)
end

"Summary dictionary describing a geometry."
function geometry_info(g::Geometry)
    chord = g.chord_over_R .* g.Rtip
    solidity = g.num_blades * trapz(g.r, chord) / (pi * g.Rtip^2)
    return Dict{String,Any}(
        "preset" => g.preset,
        "rotor_type" => g.rotor_type,
        "Rtip_m" => sig(g.Rtip),
        "Rhub_m" => sig(g.Rhub),
        "diameter_m" => sig(2 * g.Rtip),
        "num_blades" => g.num_blades,
        "precone_deg" => sig(g.precone_deg),
        "num_stations" => length(g.r),
        "airfoils" => unique(g.airfoil_names),
        "solidity" => sig(solidity, 4),
        "root_chord_m" => sig(chord[1]), "tip_chord_m" => sig(chord[end]),
        "root_twist_deg" => sig(g.twist_deg[1]), "tip_twist_deg" => sig(g.twist_deg[end]),
        "corrections" => g.corrections,
    )
end

"The geometry as the same arrays the tools accept, so results can be fed back in."
function geometry_arrays(g::Geometry)
    return Dict{String,Any}(
        "rotor_type" => g.rotor_type,
        "Rtip_m" => g.Rtip,
        "Rhub_over_Rtip" => sig(g.Rhub / g.Rtip, 5),
        "num_blades" => g.num_blades,
        "precone_deg" => g.precone_deg,
        "r_over_R" => [sig(x, 5) for x in g.r_over_R],
        "chord_over_R" => [sig(x, 5) for x in g.chord_over_R],
        "twist_deg" => [sig(x, 5) for x in g.twist_deg],
        "airfoils" => copy(g.airfoil_names),
    )
end
