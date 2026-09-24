# Built-in rotors, taken from the CCBlade documentation examples so results can be
# checked against the docs.

Base.@kwdef struct Preset
    name::String
    description::String
    rotor_type::String            # propeller | helicopter | windturbine
    Rtip::Float64                 # m
    Rhub_over_Rtip::Float64
    num_blades::Int
    precone_deg::Float64 = 0.0
    r_over_R::Vector{Float64}
    chord_over_R::Vector{Float64}
    twist_deg::Vector{Float64}
    airfoils::Vector{String}      # polar file per station
    airfoil_shape::String         # NACA 4-digit used only for the 3D visualization
    default_op::Dict{String,Any}  # operating point used when the caller gives none
end

# APC Thin Electric 10x5 (UIUC propeller database): r/R, chord/R, twist (deg)
const APC_TABLE = [
    0.15  0.130  32.76
    0.20  0.149  37.19
    0.25  0.173  33.54
    0.30  0.189  29.25
    0.35  0.197  25.64
    0.40  0.201  22.54
    0.45  0.200  20.27
    0.50  0.194  18.46
    0.55  0.186  17.05
    0.60  0.174  15.97
    0.65  0.160  14.87
    0.70  0.145  14.09
    0.75  0.128  13.39
    0.80  0.112  12.84
    0.85  0.096  12.25
    0.90  0.081  11.37
    0.95  0.061  10.19
    1.00  0.041   8.99
]
const APC_RTIP = 10 / 2 * 0.0254   # 10 inch diameter, in meters

# NREL 5 MW reference turbine (CCBlade how-to guide)
const NREL_R = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500, 28.1500, 32.2500,
                36.3500, 40.4500, 44.5500, 48.6500, 52.7500, 56.1667, 58.9000, 61.6333]
const NREL_CHORD = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748, 3.502,
                    3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
const NREL_TWIST = [13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795, 6.544,
                    5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]
const NREL_AIRFOIL_FILES = ["Cylinder1.dat", "Cylinder2.dat", "DU40_A17.dat", "DU35_A17.dat",
                            "DU30_A17.dat", "DU25_A17.dat", "DU21_A17.dat", "NACA64_A17.dat"]
const NREL_AIRFOIL_INDEX = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]
const NREL_RTIP = 63.0

const PRESETS = Dict{String,Preset}(
    "apc_10x5" => Preset(
        name = "apc_10x5",
        description = "APC Thin Electric 10x5 propeller from the UIUC database; the CCBlade tutorial case (2 blades, 0.254 m diameter, NACA 4412 polar).",
        rotor_type = "propeller",
        Rtip = APC_RTIP, Rhub_over_Rtip = 0.10, num_blades = 2,
        r_over_R = APC_TABLE[:, 1], chord_over_R = APC_TABLE[:, 2], twist_deg = APC_TABLE[:, 3],
        airfoils = fill("naca4412.dat", size(APC_TABLE, 1)),
        airfoil_shape = "4412",
        default_op = Dict{String,Any}("Vinf_m_s" => 5.0, "rpm" => 5400.0),
    ),
    "nrel_5mw" => Preset(
        name = "nrel_5mw",
        description = "NREL 5 MW reference wind turbine from the CCBlade how-to guide (3 blades, 126 m diameter, 2.5 deg precone, DU and NACA 64 airfoils, wind turbine sign conventions).",
        rotor_type = "windturbine",
        Rtip = NREL_RTIP, Rhub_over_Rtip = 1.5 / NREL_RTIP, num_blades = 3, precone_deg = 2.5,
        r_over_R = NREL_R ./ NREL_RTIP, chord_over_R = NREL_CHORD ./ NREL_RTIP, twist_deg = NREL_TWIST,
        airfoils = NREL_AIRFOIL_FILES[NREL_AIRFOIL_INDEX],
        airfoil_shape = "0021",
        default_op = Dict{String,Any}("Vinf_m_s" => 10.0, "tip_speed_ratio" => 7.55, "pitch_deg" => 0.0,
                                      "yaw_deg" => 0.0, "tilt_deg" => 5.0, "hub_height_m" => 90.0,
                                      "shear_exponent" => 0.2, "n_azimuth" => 4),
    ),
    "nasa_hover_rotor" => Preset(
        name = "nasa_hover_rotor",
        description = "Small hover rotor from the CCBlade helicopter example (Ramasamy et al., NASA): 3 untwisted constant-chord NACA 0012 blades, 0.656 m radius; set collective with pitch_deg.",
        rotor_type = "helicopter",
        Rtip = 0.656, Rhub_over_Rtip = 0.19, num_blades = 3,
        r_over_R = collect(range(0.20, 1.0, length = 30)),
        chord_over_R = fill(0.060 / 0.656, 30), twist_deg = zeros(30),
        airfoils = fill("naca0012.txt", 30),
        airfoil_shape = "0012",
        default_op = Dict{String,Any}("Vinf_m_s" => 0.0, "rpm" => 800.0, "pitch_deg" => 8.0),
    ),
)

"Summaries of the presets for the list_presets tool."
function preset_summaries()
    return [Dict{String,Any}(
                "name" => pr.name,
                "description" => pr.description,
                "rotor_type" => pr.rotor_type,
                "Rtip_m" => pr.Rtip,
                "diameter_m" => 2 * pr.Rtip,
                "Rhub_over_Rtip" => sig(pr.Rhub_over_Rtip, 4),
                "num_blades" => pr.num_blades,
                "precone_deg" => pr.precone_deg,
                "num_stations" => length(pr.r_over_R),
                "airfoils" => unique(pr.airfoils),
                "default_operating_point" => pr.default_op,
            ) for pr in (PRESETS["apc_10x5"], PRESETS["nasa_hover_rotor"], PRESETS["nrel_5mw"])]
end
