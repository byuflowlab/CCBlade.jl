# airfoil methods

using Printf: @printf

export afeval, AFType, SimpleAF, AlphaAF, AlphaMachAF, AlphaReAF, AlphaReMachAF
export read_AlphaAF, read_AlphaReAF, read_AlphaMachAF, read_AlphaReMachAF, write_af
export mach_correction, MachCorrection, PrandtlGlauert
export re_correction, ReCorrection, SkinFriction, LaminarSkinFriction, TurbulentSkinFriction
export rotation_correction, RotationCorrection, DuSeligEggers
export tip_correction, TipCorrection, PrandtlTip, PrandtlTipHub
export viterna

# TODO: check for repeat alpha as that will mess up fit.

############# file parsing ##############
"""
A basic airfoil file format.  `nheader` is the number of header lines,
which will be skipped.  For one Reynolds/Mach number.
Additional data like cm is optional but will be ignored.

format:

informational header\n
Re\n
Mach\n
alpha1 cl1 cd1 ...\n
alpha2 cl2 cd2\n
alpha3 cl3 cd3\n
...
"""
function parsefile(filename, radians)

    alpha = Float64[]
    cl = Float64[]
    cd = Float64[]
    info = ""
    Re = 1.0
    Mach = 1.0

    open(filename) do f

        # skip header
        info = readline(f)
        Re = parse(Float64, readline(f))
        Mach = parse(Float64, readline(f))
        
        for line in eachline(f)
            parts = split(line)
            push!(alpha, parse(Float64, parts[1]))
            push!(cl, parse(Float64, parts[2]))
            push!(cd, parse(Float64, parts[3]))
        end    
    end

    if !radians
        alpha *= pi/180
    end
    
    return info, Re, Mach, alpha, cl, cd
end


function writefile(filename, info, Re, Mach, alpha, cl, cd, radians)
    open(filename, "w") do f
        @printf(f, "%s\n", info)
        @printf(f, "%.17g\n", Re)
        @printf(f, "%.17g\n", Mach)

        factor = 1.0
        if !radians
            factor = 180/pi
        end

        for i = 1:length(alpha)
            @printf(f, "%.17g\t%.17g\t%.17g\n", alpha[i]*factor, cl[i], cd[i])
        end
    end

    return nothing
end
###################################

########### Airfoil Data Types ###############

# -- interface -----
abstract type AFType end

"""
    afeval(af::AFType, alpha, Re, Mach)

Evaluate airfoil aerodynamic performance

**Arguments**
- `af::AFType or Function`: dispatch on AFType or if function call: `cl, cd = af(alpha, Re, Mach)`
- `alpha::Float64`: angle of attack in radians
- `Re::Float64`: Reynolds number
- `Mach::Float64`: Mach number

**Returns**
- `cl::Float64`: lift coefficient
- `cd::Float64`: drag coefficient
"""
function afeval(af::AFType, alpha, Re, Mach)
    @error "need to choose an AFType"
end
# ---------------

# --- Function Type -----
function afeval(af::Function, alpha, Re, Mach)
    return af(alpha, Re, Mach)
end
# ---------------

# --- Simple paramterization ------  

"""
    SimpleAF(m, alpha0, clmax, clmin, cd0, cd2)

A simple parameterized lift and drag curve.  
- `cl = m (alpha - alpha0)` (capped by clmax/clmin)
- `cd = cd0 + cd2 * cl^2`

**Arguments**
- `m::Float64`: lift curve slope
- `alpha0::Float64`: zero-lift angle of attack
- `clmax::Float64`: maximum lift coefficient
- `clmin::Float64`: minimum lift coefficient
- `cd0::Float64`: zero lift drag
- `cd2::Float64`: quadratic drag term
"""
struct SimpleAF{TF} <: AFType
    m::TF
    alpha0::TF
    clmax::TF
    clmin::TF
    cd0::TF
    cd2::TF
end

function afeval(af::SimpleAF, alpha, Re, Mach)
    cl = af.m*(alpha - af.alpha0)
    cl = min(cl, af.clmax)
    cl = max(cl, af.clmin)

    cd = af.cd0 + af.cd2*cl^2

    return cl, cd
end

# ---------------

# --- Spline Type (just alpha)

"""
    AlphaAF(alpha, cl, cd, info, Re, Mach)
    AlphaAF(alpha, cl, cd, info, Re=0.0, Mach=0.0)
    AlphaAF(alpha, cl, cd, info="CCBlade generated airfoil", Re=0.0, Mach=0.0)
    AlphaAF(filename::String; radians=true)

Airfoil data that varies with angle of attack.  Data is fit with an Akima spline.

**Arguments**
- `alpha::Vector{Float64}`: angles of attack
- `cl::Vector{Float64}`: corresponding lift coefficients
- `cd::Vector{Float64}`: corresponding drag coefficients
- `info::String`: a description of this airfoil data (just informational)
- `Re::Float64`: Reynolds number data was taken at (just informational)
- `Mach::Float64`: Mach number data was taken at (just informational)

or

a file

**Arguments**
- `filename::String`: name/path of file to read in
- `radians::Bool`: true if angle of attack in file is given in radians
"""
struct AlphaAF{TF, TS, TSP} <: AFType 
    alpha::Vector{TF}
    cl::Vector{TF}
    cd::Vector{TF}
    info::TS # not used except for info in file
    Re::TF  # not used except for info in file
    Mach::TF # not used except for info in file
    clspline::TSP  # cache these for speed
    cdspline::TSP
end


function AlphaAF(alpha, cl, cd, info, Re, Mach)
    afcl = FLOWMath.Akima(alpha, cl)
    afcd = FLOWMath.Akima(alpha, cd)

    return AlphaAF(alpha, cl, cd, info, Re, Mach, afcl, afcd)
end

AlphaAF(alpha, cl, cd, info) = AlphaAF(alpha, cl, cd, info, 0.0, 0.0)
AlphaAF(alpha, cl, cd) = AlphaAF(alpha, cl, cd, "CCBlade generated airfoil", 0.0, 0.0)

function AlphaAF(filename::String; radians=true)
    info, Re, Mach, alpha, cl, cd = parsefile(filename, radians)
    return AlphaAF(alpha, cl, cd, info, Re, Mach)
end


function afeval(af::AlphaAF, alpha, Re, Mach)
    return af.clspline(alpha), af.cdspline(alpha)
end


"""
    write_af(filename(s), af::AFType; radians=true)

Write airfoil data to file

**Arguments**
- `filename(s)::String or Vector{String} or Matrix{String}`: name/path of file to write to
- `af::AFType`: writing is dispatched based on type (AlphaAF, AlphaReAF, etc.)
- `radians::Bool`: true if you want angle of attack to be written in radians
"""
function write_af(filename, af::AlphaAF; radians=true)
    writefile(filename, af.info, af.Re, af.Mach, af.alpha, af.cl, af.cd, radians)
    return nothing
end
# ---------------


# ----- higher order splines

"""
    AlphaReAF(alpha, Re, cl, cd, info, Mach)
    AlphaReAF(alpha, Re, cl, cd, info)
    AlphaReAF(alpha, Re, cl, cd)
    read_AlphaReAF(filenames::Vector{String}; radians=true)

Airfoil data that varies with angle of attack and Reynolds number.  
Data is fit with a recursive Akima spline.

**Arguments**
- `alpha::Vector{Float64}`: angles of attack
- `Re::Vector{Float64}`: Reynolds numbers
- `cl::Matrix{Float64}`: lift coefficients where cl[i, j] corresponds to alpha[i], Re[j]
- `cd::Matrix{Float64}`: drag coefficients where cd[i, j] corresponds to alpha[i], Re[j]
- `info::String`: a description of this airfoil data (just informational)
- `Mach::Float64`: Mach number data was taken at (just informational)

or

filenames with one file per Reynolds number.

**Arguments**
- `filenames::Vector{String}`: name/path of files to read in, each at a different Reynolds number in ascending order
- `radians::Bool`: true if angle of attack in file is given in radians
"""
struct AlphaReAF{TF, TS} <: AFType 
    alpha::Vector{TF}
    Re::Vector{TF}
    cl::Matrix{TF}
    cd::Matrix{TF}
    info::TS # not used except for info in file
    Mach::TF # not used except for info in file
end

AlphaReAF(alpha, Re, cl, cd, info) = AlphaReAF(alpha, Re, cl, cd, info, 0.0)
AlphaReAF(alpha, Re, cl, cd) = AlphaReAF(alpha, Re, cl, cd, "CCBlade generated airfoil", 0.0)

function AlphaReAF(filenames::Vector{String}; radians=true)
    
    info, Re1, Mach, alpha, cl1, cd1 = parsefile(filenames[1], radians)  # assumes common alpha across files, also common info and common Mach
    nalpha = length(alpha)
    ncond = length(filenames)

    cl = Array{Float64}(undef, nalpha, ncond)
    cd = Array{Float64}(undef, nalpha, ncond)
    Re = Array{Float64}(undef, ncond)
    cl[:, 1] = cl1
    cd[:, 1] = cd1
    Re[1] = Re1

    # iterate over remaining files
    for i = 2:ncond
        _, Rei, _, _, cli, cdi = parsefile(filenames[i], radians)
        cl[:, i] = cli
        cd[:, i] = cdi
        Re[i] = Rei
    end
    
    return AlphaReAF(alpha, Re, cl, cd, info, Mach)
end

function afeval(af::AlphaReAF, alpha, Re, Mach)

    cl = FLOWMath.interp2d(FLOWMath.akima, af.alpha, af.Re, af.cl, [alpha], [Re])[1]
    cd = FLOWMath.interp2d(FLOWMath.akima, af.alpha, af.Re, af.cd, [alpha], [Re])[1]

    return cl, cd
end

function write_af(filenames, af::AlphaReAF; radians=true)
    for i = 1:length(af.Re)
        writefile(filenames[i], af.info, af.Re[i], af.Mach, af.alpha, af.cl[:, i], af.cd[:, i], radians)
    end
    return nothing
end

# ---------------

"""
    AlphaMachAF(alpha, Mach, cl, cd, info, Re)
    AlphaMachAF(alpha, Mach, cl, cd, info)
    AlphaMachAF(alpha, Mach, cl, cd)
    AlphaMachAF(filenames::Vector{String}; radians=true)

Airfoil data that varies with angle of attack and Mach number.  
Data is fit with a recursive Akima spline.

**Arguments**
- `alpha::Vector{Float64}`: angles of attack
- `Mach::Vector{Float64}`: Mach numbers
- `cl::Matrix{Float64}`: lift coefficients where cl[i, j] corresponds to alpha[i], Mach[j]
- `cd::Matrix{Float64}`: drag coefficients where cd[i, j] corresponds to alpha[i], Mach[j]
- `info::String`: a description of this airfoil data (just informational)
- `Re::Float64`: Reynolds number data was taken at (just informational)

or

filenames with one file per Mach number.

**Arguments**
- `filenames::Vector{String}`: name/path of files to read in, each at a different Mach number in ascending order
- `radians::Bool`: true if angle of attack in file is given in radians
"""
struct AlphaMachAF{TF, TS} <: AFType 
    alpha::Vector{TF}
    Mach::Vector{TF}
    cl::Matrix{TF}
    cd::Matrix{TF}
    info::TS # not used except for info in file
    Re::TF # not used except for info in file
end

AlphaMachAF(alpha, Mach, cl, cd, info) = AlphaMachAF(alpha, Mach, cl, cd, info, 0.0)
AlphaMachAF(alpha, Mach, cl, cd) = AlphaMachAF(alpha, Mach, cl, cd, "CCBlade generated airfoil", 0.0)


function AlphaMachAF(filenames::Vector{String}; radians=true)
    
    info, Re, Mach1, alpha, cl1, cd1 = parsefile(filenames[1], radians)  # assumes common alpha across files, also common info and common Re
    nalpha = length(alpha)
    ncond = length(filenames)

    cl = Array{Float64}(undef, nalpha, ncond)
    cd = Array{Float64}(undef, nalpha, ncond)
    Mach = Array{Float64}(undef, ncond)
    cl[:, 1] = cl1
    cd[:, 1] = cd1
    Mach[1] = Mach1

    # iterate over remaining files
    for i = 2:ncond
        _, _, Machi, _, cli, cdi = parsefile(filenames[i], radians)
        cl[:, i] = cli
        cd[:, i] = cdi
        Mach[i] = Machi
    end
    
    return AlphaMachAF(alpha, Mach, cl, cd, info, Re)
end

function afeval(af::AlphaMachAF, alpha, Re, Mach)

    cl = FLOWMath.interp2d(FLOWMath.akima, af.alpha, af.Mach, af.cl, [alpha], [Mach])[1]
    cd = FLOWMath.interp2d(FLOWMath.akima, af.alpha, af.Mach, af.cd, [alpha], [Mach])[1]

    return cl, cd
end


function write_af(filenames, af::AlphaMachAF; radians=true)
    for i = 1:length(af.Mach)
        writefile(filenames[i], af.info, af.Re, af.Mach[i], af.alpha, af.cl[:, i], af.cd[:, i], radians)
    end
    return nothing
end

# ---------------

"""
    AlphaReMachAF(alpha, Re, Mach, cl, cd, info)
    AlphaReMachAF(alpha, Re, Mach, cl, cd)
    AlphaReMachAF(filenames::Matrix{String}; radians=true)

Airfoil data that varies with angle of attack, Reynolds number, and Mach number.  
Data is fit with a recursive Akima spline.

**Arguments**
- `alpha::Vector{Float64}`: angles of attack
- `Re::Vector{Float64}`: Reynolds numbers
- `Mach::Vector{Float64}`: Mach numbers
- `cl::Array{Float64}`: lift coefficients where cl[i, j, k] corresponds to alpha[i], Re[j], Mach[k]
- `cd::Array{Float64}`: drag coefficients where cd[i, j, k] corresponds to alpha[i], Re[j], Mach[k]
- `info::String`: a description of this airfoil data (just informational)

or files with one per Re/Mach combination

**Arguments**
- `filenames::Matrix{String}`: name/path of files to read in.  filenames[i, j] corresponds to Re[i] Mach[j] with Reynolds number and Mach number in ascending order.
- `radians::Bool`: true if angle of attack in file is given in radians
"""
struct AlphaReMachAF{TF, TS} <: AFType 
    alpha::Vector{TF}
    Re::Vector{TF}
    Mach::Vector{TF}
    cl::Array{TF}
    cd::Array{TF}
    info::TS
end

AlphaReMachAF(alpha, Re, Mach, cl, cd) = AlphaReMachAF(alpha, Re, Mach, cl, cd, "CCBlade generated airfoil")

function AlphaReMachAF(filenames::Matrix{String}; radians=true)

    info, _, _, alpha, _, _ = parsefile(filenames[1, 1], radians)  # assumes common alpha and info across files
    nalpha = length(alpha)
    nRe, nMach = size(filenames)
    
    cl = Array{Float64}(undef, nalpha, nRe, nMach)
    cd = Array{Float64}(undef, nalpha, nRe, nMach)
    Re = Array{Float64}(undef, nRe)
    Mach = Array{Float64}(undef, nMach)

    for j = 1:nMach
        for i = 1:nRe
            _, Rei, Machj, _, clij, cdij = parsefile(filenames[i, j], radians)
            cl[:, i, j] = clij
            cd[:, i, j] = cdij
            Re[i] = Rei
            Mach[j] = Machj  # NOTE: probably should add check to prevent user error here.
        end
    end

    return AlphaReMachAF(alpha, Re, Mach, cl, cd, info)
end

function afeval(af::AlphaReMachAF, alpha, Re, Mach)
    cl = FLOWMath.interp3d(FLOWMath.akima, af.alpha, af.Re, af.Mach, af.cl, [alpha], [Re], [Mach])[1]
    cd = FLOWMath.interp3d(FLOWMath.akima, af.alpha, af.Re, af.Mach, af.cd, [alpha], [Re], [Mach])[1]
    
    return cl, cd
end


function write_af(filenames, af::AlphaReMachAF; radians=true)
    nre = length(af.Re)
    nm = length(af.Mach)

    for i = 1:nre
        for j = 1:nm
            writefile(filenames[i, j], af.info, af.Re[i], af.Mach[j], af.alpha, af.cl[:, i, j], af.cd[:, i, j], radians)
        end
    end

    return nothing
end

#####################################




########## Mach Corrections ################

abstract type MachCorrection end

struct PrandtlGlauert <: MachCorrection end

"""
    mach_correction(::MachCorrection, cl, cd, Mach)

Mach number correction for lift/drag coefficient

**Arguments**
- `mc::MachCorrection`: used for dispatch
- `cl::Float64`: lift coefficient before correction
- `cd::Float64`: drag coefficient before correction
- `Mach::Float64`: Mach number

**Returns**
- `cl::Float64`: lift coefficient after correction
- `cd::Float64`: drag coefficient after correction
"""
function mach_correction(::MachCorrection, cl, cd, Mach)
    return cl, cd
end

"""
    mach_correction(::PrandtlGlauert, cl, cd, Mach)

Prandtl/Glauert Mach number correction for lift coefficient
"""
function mach_correction(::PrandtlGlauert, cl, cd, Mach)
    beta = sqrt(1 - Mach^2)
    cl /= beta
    return cl, cd
end

########## Reynolds corrections ##############

abstract type ReCorrection end

"""
    re_correction(re::ReCorrection, cl, cd, Re)

Reynolds number correction for lift/drag coefficient

**Arguments**
- `re::ReCorrection`: used for dispatch
- `cl::Float64`: lift coefficient before correction
- `cd::Float64`: drag coefficient before correction
- `Re::Float64`: Reynolds number

**Returns**
- `cl::Float64`: lift coefficient after correction
- `cd::Float64`: drag coefficient after correction
"""
function re_correction(::ReCorrection, cl, cd, Re)
    return cl, cd
end


"""
    SkinFriction(Re0, p)

Skin friction model for a flat plate.
`cd *= (Re0 / Re)^p`

**Arguments**
- `Re0::Float64`: reference Reynolds number (i.e., no corrections at this number)
- `p::Float64`: exponent in flat plate model.  0.5 for laminar (Blasius solution), ~0.2 for fully turbulent (Schlichting empirical fit)
"""
struct SkinFriction{TF} <: ReCorrection 
    Re0::TF 
    p::TF  
end
LaminarSkinFriction(Re0) = SkinFriction(Re0, 0.5)
TurbulentSkinFriction(Re0) = SkinFriction(Re0, 0.2)

"""
    re_correction(sf::SkinFriction, cl, cd, Re)

Skin friction coefficient correction based on flat plat drag increases with Reynolds number.
"""
function re_correction(sf::SkinFriction, cl, cd, Re)
    cd *= (sf.Re0 / Re)^sf.p
    return cl, cd
end

############ Rotation Correction ###################

abstract type RotationCorrection end

"""
    rotation_correction(rc::RotationCorrection, cl, cd, cr, rR, tsr, alpha, phi=alpha, alpha_max_corr=30*pi/180)

Rotation correction (3D stall delay).

**Arguments**
- `rc::RotationCorrection`: used for dispatch
- `cl::Float64`: lift coefficient before correction
- `cd::Float64`: drag coefficient before correction
- `cr::Float64`: local chord / local radius
- `rR::Float64`: local radius / tip radius
- `tsr::Float64`: local tip speed ratio (Omega r / Vinf)
- `alpha::Float64`: local angle of attack
- `phi::Float64`: local inflow angles (defaults to angle of attack is precomputing since it is only known for on-the-fly computations)
- `alpha_max_corr::Float64`: angle of attack for maximum correction (tapers off to zero by 90 degrees)

**Returns**
- `cl::Float64`: lift coefficient after correction
- `cd::Float64`: drag coefficient after correction
"""
function rotation_correction(rc::RotationCorrection, cl, cd, cr, rR, tsr, alpha, phi=alpha, alpha_max_corr=30*pi/180)
    return cl, cd
end

"""
    DuSeligEggers(a, b, d, m, alpha0)
    DuSeligEggers(a=1.0, b=1.0, d=1.0, m=2*pi, alpha0=0.0)  # uses defaults

DuSelig correction for lift an Eggers correction for drag.

**Arguments**
- `a, b, d::Float64`: parameters in Du-Selig paper.  Normally just 1.0 for each.
- `m::Float64`: lift curve slope.  Defaults to 2 pi for zero argument version.
- `alpha0::Float64`: zero-lift angle of attack.  Defaults to 0 for zero argument version.
"""
struct DuSeligEggers{TF} <: RotationCorrection
    a::TF
    b::TF
    d::TF
    m::TF
    alpha0::TF
end

DuSeligEggers() = DuSeligEggers(1.0, 1.0, 1.0, 2*pi, 0.0)


function rotation_correction(du::DuSeligEggers, cl, cd, cr, rR, tsr, alpha, phi=alpha, alpha_max_corr=30*pi/180)
    
    # Du-Selig correction for lift
    Lambda = tsr / sqrt(1 + tsr^2)
    expon = du.d / (Lambda * rR)
    fcl = 1.0/du.m*(1.6*cr/0.1267*(du.a-cr^expon)/(du.b+cr^expon)-1)

    # linear lift line
    cl_linear = du.m*(alpha - du.alpha0)

    # adjustment for max correction
    amax = atan(1/0.12) - 5*pi/180  # account for singularity in Eggers (not pi/2)
    if abs(alpha) >= amax 
        adj = 0.0
    elseif abs(alpha) > alpha_max_corr
        adj = ((amax-abs(alpha))/(amax-alpha_max_corr))^2
    else
        adj = 1.0
    end
    
    # increment in cl
    deltacl = adj*fcl*(cl_linear - cl)
    cl += deltacl

    # Eggers correction for drag
    deltacd = deltacl * (sin(phi) - 0.12*cos(phi))/(cos(phi) + 0.12*sin(phi))  # note that we can actually use phi instead of alpha as is done in airfoilprep.py b/c this is done at each iteration
    cd += deltacd

    return cl, cd
end    


# ------ Extract Linear Portion of Curve -----------



function linearliftcoeff(af::T, Re, Mach) where T <: Union{AlphaAF, AlphaReAF, AlphaMachAF, AlphaReMachAF}

    clwrap(alpha) = afeval(af, alpha, Re, Mach)[1]
    alpha = range(-2*pi/180, 5*pi/180, length=20)
    cl = clwrap.(alpha)
    
    return linearliftregression(alpha, cl)
end

function linearliftcoeff(alpha, cl, alphamin=-2*pi/180, alphamax=5*pi/180)

    idxmin = findfirst(alpha .> alphamin)
    idxmax = findfirst(alpha .> alphamax)
    alphasub = @view alpha[idxmin:idxmax]
    clsub = @view cl[idxmin:idxmax]
    return linearliftregression(alphasub, clsub)
end

function linearliftregression(alphasub, clsub)
    # linear regression
    coeff = [alphasub ones(length(alphasub))] \ clsub
    m = coeff[1]  # lift curve slope
    alpha0 = -coeff[2] / m  # zero-lift angle of attack

    return m, alpha0
end





############## tip correction ################

abstract type TipCorrection end 

"""
    tip_correction(::TipCorrection, r, Rhub, Rtip, phi, B)

Tip corrections for 3D flow.

**Arguments**
- `tc::TipCorrection`: used for dispatch
- `r::Float64`: local radius
- `Rhub::Float64`: hub radius
- `Rtip::Float64`: tip radius
- `phi::Float64`: inflow angle
- `B::Integer`: number of blades

**Returns**
- `F::Float64`: tip loss factor to multiple against loads.
"""
function tip_correction(::TipCorrection, cl, cd, Mach)
    return 1.0
end

"""
    PrandtlTip()

Standard Prandtl tip loss correction.
"""
struct PrandtlTip <: TipCorrection end

"""
    PrandtlTipHub()

Standard Prandtl tip loss correction plus hub loss correction of same form.
"""
struct PrandtlTipHub <: TipCorrection end

function tip_correction(::PrandtlTip, r, Rhub, Rtip, phi, B)
    
    asphi = abs(sin(phi))
    factortip = B/2.0*(Rtip/r - 1)/asphi
    F = 2.0/pi*acos(exp(-factortip))

    return F
end

function tip_correction(::PrandtlTipHub, r, Rhub, Rtip, phi, B)

    # Prandtl's tip and hub loss factor
    asphi = abs(sin(phi))
    factortip = B/2.0*(Rtip/r - 1)/asphi
    Ftip = 2.0/pi*acos(exp(-factortip))
    factorhub = B/2.0*(r/Rhub - 1)/asphi
    Fhub = 2.0/pi*acos(exp(-factorhub))
    F = Ftip * Fhub

    return F
end

############## Extrapolation ##################

"""
    viterna(alpha, cl, cd, cr75, nalpha=50)

Viterna extrapolation.  Follows Viterna paper and somewhat follows NREL version of AirfoilPrep, but with some modifications for better robustness and smoothness.

**Arguments**
- `alpha::Vector{Float64}`: angles of attack
- `cl::Vector{Float64}`: correspnding lift coefficients
- `cd::Vector{Float64}`: correspnding drag coefficients
- `cr75::Float64`: chord/Rtip at 75% Rtip
- `nalpha::Int64`: number of discrete points (angles of attack) to include in extrapolation

**Returns**
- `alpha::Vector{Float64}`: angle of attack from -pi to pi
- `cl::Vector{Float64}`: correspnding extrapolated lift coefficients
- `cd::Vector{Float64}`: correspnding extrapolated drag coefficients
"""
function viterna(alpha, cl, cd, cr75, nalpha=50)

    # estimate cdmax
    AR = 1.0 / cr75  
    cdmaxAR = 1.11 + 0.018*AR
    cdmax = max(maximum(cd), cdmaxAR)

    # find clmax
    i_ps = argmax(cl)  # positive stall
    cl_ps = cl[i_ps]
    cd_ps = cd[i_ps]
    a_ps = alpha[i_ps]

    # and clmin
    i_bs = alpha .< a_ps  # before stall
    i_ns = argmin(cl[i_bs])  # negative stall
    cl_ns = cl[i_bs][i_ns]
    cd_ns = cd[i_bs][i_ns]
    a_ns = alpha[i_bs][i_ns]

    # coefficients in method
    B1pos = cdmax
    A1pos = B1pos/2.0 * ones(nalpha)
    sa = sin(a_ps)
    ca = cos(a_ps)
    A2pos = (cl_ps - cdmax*sa*ca)*sa/ca^2
    B2pos = (cd_ps - cdmax*sa^2)/ca * ones(nalpha)

    B1neg = cdmax
    A1neg = B1neg/2.0
    sa = sin(a_ns)
    ca = cos(a_ns)
    A2neg = (cl_ns - cdmax*sa*ca)*sa/ca^2 * ones(nalpha)
    B2neg = (cd_ns - cdmax*sa^2)/ca * ones(nalpha)
    
    # angles of attack to extrapolate to
    apos = range(alpha[end], pi, length=nalpha+1)
    apos = apos[2:end]  # don't duplicate point
    aneg = range(-pi, alpha[1], length=nalpha+1)
    aneg = aneg[1:end-1]  # don't duplicate point

    # high aoa adjustments
    adjpos = ones(nalpha)
    idx = findall(apos .>= pi/2)
    adjpos[idx] .= -0.7
    A1pos[idx] .*= -1
    B2pos[idx] .*= -1

    # idx = findall(aneg .<= -alpha[end])
    
    adjneg = ones(nalpha)
    idx = findall(aneg .<= -pi/2)
    adjneg[idx] .= 0.7
    A2neg[idx] .*= -1
    B2neg[idx] .*= -1

    # extrapolate 
    clpos = @. adjpos * (A1pos*sin(2*apos) + A2pos*cos(apos)^2/sin(apos))
    cdpos = @. B1pos*sin(apos)^2 + B2pos*cos(apos)
    clneg = @. adjneg * (A1neg*sin(2*aneg) + A2neg*cos(aneg)^2/sin(aneg))
    cdneg = @. B1neg*sin(aneg)^2 + B2neg*cos(aneg)

    # # override region between -alpha_high and alpha_low (if it exists)
    # idx = findall(-alpha[end] .<= aneg .<= alpha[1])
    # @. clneg[idx] = -cl[end]*0.7 + (aneg[idx]+alpha[end])/(alpha[1]+alpha[end])*(cl[1]+cl[end]*0.7)
    # @. cdneg[idx] = cd[1] + (aneg[idx]-alpha[1])/(-alpha[end]-alpha[1])*(cd[end]-cd[1])


    # override with linear variation at ends
    idx = findall(apos .>= pi-a_ps)
    @. clpos[idx] = (apos[idx] - pi)/a_ps*cl_ps*0.7  
    idx = findall(aneg .<= -pi-a_ns)
    @. clneg[idx] = (aneg[idx] + pi)/a_ns*cl_ns*0.7  

    # concatenate
    alphafull = [aneg; alpha; apos]
    clfull = [clneg; cl; clpos]
    cdfull = [cdneg; cd; cdpos]

    # don't allow negative drag
    cdfull = max.(cdfull, 0.0001)
    return alphafull, clfull, cdfull
end


