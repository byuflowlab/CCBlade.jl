# Correction Methods 

export NoMachCorrection, PrandtlGlauert, machcorrection
export NoReCorrection, SkinFriction, recorrection
export NoRotationCorrection, DuSeligEggers, rotationcorrection
export NoTipCorrection, PrandtlTipOnly, Prandtl, tipcorrection
export linearliftcoeff

# ------ Mach ---------

abstract type MachCorrection end

struct NoMachCorrection <: MachCorrection end
struct PrandtlGlauert <: MachCorrection end

function machcorrection(::NoMachCorrection, cl, cd, Mach)
    return cl, cd
end

function machcorrection(::PrandtlGlauert, cl, cd, Mach)
    beta = sqrt(1 - Mach^2)
    cl /= beta
    return cl, cd
end


# --------- Reynolds number ------------

abstract type ReCorrection end

struct NoReCorrection <: ReCorrection end
struct SkinFriction{TF} <: ReCorrection 
    Re0::TF  # reference reynolds number
    p::TF  # exponent.  approx 0.2 fully turbulent (Schlichting), 0.5 fully laminar (Blasius)
end

function recorrection(::NoReCorrection, cl, cd, Re)
    return cl, cd
end

function recorrection(sf::SkinFriction, cl, cd, Re)
    cd *= (sf.Re0 / Re)^sf.p
    return cl, cd
end



# ------------ Rotation -----------------

abstract type RotationCorrection end

struct NoRotationCorrection <: RotationCorrection end
struct DuSeligEggers{TF} <: RotationCorrection
    a::TF
    b::TF
    d::TF
end
DuSeligEggers() = DuSeligEggers(1.0, 1.0, 1.0)

function rotationcorrection(::NoRotationCorrection, cl, cd, cr, rR, tsr, alpha, m, alpha0, phi)
    return cl, cd
end

function rotationcorrection(du::DuSeligEggers, cl, cd, cr, rR, tsr, alpha, m, alpha0, phi)
    # Du-Selig correction for lift
    Lambda = tsr / sqrt(1 + tsr^2)
    expon = du.d / (Lambda * rR)
    fcl = 1.0/m*(1.6*cr/0.1267*(du.a-cr^expon)/(du.b+cr^expon)-1)
    cl_linear = m*(alpha - alpha0)
    deltacl = fcl*(cl_linear - cl)
    cl += deltacl

    # Eggers correction for drag
    deltacd = deltacl*(sin(phi) - 0.12*cos(phi))/(cos(phi) + 0.12*sin(phi))  # note that we can actually use phi instead of alpha as is done in airfoilprep.py b/c this is done at each iteration
    cd += deltacd

    return cl, cd
end    

function linearliftcoeff(alphasub, clsub)
    # linear regression
    coeff = [alphasub ones(length(alphasub))] \ clsub
    m = coeff[1]  # lift curve slope
    alpha0 = -coeff[2] / m  # zero-lift angle of attack

    return m, alpha0
end

# ------------ tip correction  ----------------

abstract type TipCorrection end 

struct NoTipCorrection <: TipCorrection end
struct PrandtlTipOnly <: TipCorrection end
struct Prandtl <: TipCorrection end

function tiplossfactor(::NoTipCorrection, r, Rhub, Rtip, phi, B)
    return 1.0
end

function tiplossfactor(::PrandtlTipOnly, r, Rhub, Rtip, phi, B)
    
    asphi = abs(sin(phi))
    factortip = B/2.0*(Rtip/r - 1)/asphi
    F = 2.0/pi*acos(exp(-factortip))

    return F
end

function tiplossfactor(::Prandtl, r, Rhub, Rtip, phi, B)

    # Prandtl's tip and hub loss factor
    asphi = abs(sin(phi))
    factortip = B/2.0*(Rtip/r - 1)/asphi
    Ftip = 2.0/pi*acos(exp(-factortip))
    factorhub = B/2.0*(r/Rhub - 1)/asphi
    Fhub = 2.0/pi*acos(exp(-factorhub))
    F = Ftip * Fhub

    return F
end
