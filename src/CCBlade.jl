#=
Author: Andrew Ning

A general blade element momentum (BEM) method for propellers/fans and turbines.

Some unique features:
- a simple yet very robust solution method
- allows for non-ideal conditions like reversed flow, or free rotation
- allows arbitrary inflow
- convenience methods for common wind turbine inflow scenarios

=#

module CCBlade

import FLOWMath

export Rotor, Section, OperatingPoint, Outputs
export af_from_files, af_from_data
export simple_op, windturbine_op
export solve, thrusttorque, nondim


include("derivatives.jl")
include("openmdao.jl")

# --------- structs -------------


"""
    Rotor(Rhub, Tip, B, turbine, pitch, precone)

Scalar parameters defining the rotor.  

**Arguments**
- `Rhub::Float64`: hub radius (along blade length)
- `Rtip::Float64`: tip radius (along blade length)
- `B::Int64`: number of blades
- `turbine::Bool`: true if turbine, false if propeller
- `pitch::Float64`: pitch angle (rad).  defined same direction as twist.
- `precone::Float64`: precone angle
"""
mutable struct Rotor{TF, TI, TB}
    
    Rhub::TF
    Rtip::TF
    B::TI
    turbine::TB
    pitch::TF
    precone::TF

end

# convenience constructor for no precone and no pitch
Rotor(Rhub, Rtip, B, turbine) = Rotor(Rhub, Rtip, B, turbine, zero(Rhub), zero(Rhub))
Rotor(Rhub, Rtip, B, turbine, pitch) = Rotor(Rhub, Rtip, B, turbine, pitch, zero(Rhub))

"""
    Section(r, chord, theta, af)

Define sectional properties for one station along rotor
    
**Arguments**
- `r::Float64`: radial location along turbine (Rhub < r < Rtip)
- `chord::Float64`: corresponding local chord length
- `theta::Float64`: corresponding twist angle (radians)
- `af::function`: a function of the form: cl, cd = af(alpha, Re, Mach)
"""
mutable struct Section{TF, TAF}
    
    r::TF
    chord::TF
    theta::TF
    af::TAF

end

# make rotor broadcastable as a single entity
# Base.Broadcast.broadcastable(r::Rotor) = Ref(r) 

# convenience function to access fields within an array of structs
function Base.getproperty(obj::Vector{Section{TF, TAF}}, sym::Symbol) where {TF, TAF}
    return getfield.(obj, sym)
end


"""
    OperatingPoint(Vx, Vy, rho, mu=1.0, asound=1.0)

Operation point for a rotor.  The x direction is the axial direction, and y direction is the tangential direction in the rotor plane.  See Documentation for more detail on coordinate systems.

**Arguments**
- `Vx::Float64`: velocity in x-direction along blade
- `Vy::Float64`: velocity in y-direction along blade
- `rho::Float64`: fluid density
- `mu::Float64`: fluid dynamic viscosity (unused if Re not included in airfoil data)
- `asound::Float64`: fluid speed of sound (unused if Mach not included in airfoil data)
"""
mutable struct OperatingPoint{TF, TF2}
    Vx::TF
    Vy::TF
    rho::TF2  # different type to accomodate ReverseDiff
    mu::TF2
    asound::TF2
end

# convenience constructor when Re and Mach are not used.
OperatingPoint(Vx, Vy, rho) = OperatingPoint(Vx, Vy, rho, one(rho), one(rho)) 

# convenience function to access fields within an array of structs
function Base.getproperty(obj::Vector{OperatingPoint{TF, TF2}}, sym::Symbol) where {TF, TF2}
    return getfield.(obj, sym)
end


"""
    Outputs(Np, Tp, a, ap, u, v, phi, W, cl, cd, F)

Outputs from the BEM solver.

**Arguments**
- `Np::Vector{Float64}`: normal force per unit length
- `Tp::Vector{Float64}`: tangential force per unit length
- `a::Vector{Float64}`: axial induction factor
- `ap::Vector{Float64}`: tangential induction factor
- `u::Vector{Float64}`: axial induced velocity
- `v::Vector{Float64}`: tangential induced velocity
- `phi::Vector{Float64}`: inflow angle
- `alpha::Vector{Float64}`: angle of attack
- `W::Vector{Float64}`: inflow velocity
- `cl::Vector{Float64}`: lift coefficient
- `cd::Vector{Float64}`: drag coefficient
- `cn::Vector{Float64}`: normal force coefficient
- `ct::Vector{Float64}`: tangential force coefficient
- `F::Vector{Float64}`: hub/tip loss correction
- `G::Vector{Float64}`: effective hub/tip loss correction for induced velocities: u = Vx * a * G, v = Vy * ap * G
"""
struct Outputs{TF}

    Np::TF
    Tp::TF
    a::TF
    ap::TF
    u::TF
    v::TF
    phi::TF
    alpha::TF
    W::TF
    cl::TF
    cd::TF
    cn::TF
    ct::TF
    F::TF
    G::TF

end

# convenience constructor to initialize
Outputs() = Outputs(0.0, 0.0, 0.0, 0.0, 0.1, 0.1, -5.0*pi/180.0, 0.0, 0.0, 0.0, 0.0)

# convenience function to access fields within an array of structs
function Base.getproperty(obj::Vector{Outputs{TF}}, sym::Symbol) where TF
    return getfield.(obj, sym)
end


# -------------------------------



# ----------- airfoil ---------------


"""
    parse_af_file(filename, header=1)

Private function. Read an airfoil file.  For one Reynolds/Mach number.
Additional data like cm is optional but will be ignored.
alpha should be in degrees

format:

header\n
alpha1 cl1 cd1 ...\n
alpha2 cl2 cd2\n
alpha3 cl3 cd3\n
...

Returns arrays for alpha (in radians), cl, cd
"""
function parse_af_file(filename, header=1)

    alpha = Float64[]
    cl = Float64[]
    cd = Float64[]

    open(filename) do f

        # skip header
        for i = 1:header
            readline(f)
        end

        for line in eachline(f)
            parts = split(line)
            push!(alpha, parse(Float64, parts[1]))
            push!(cl, parse(Float64, parts[2]))
            push!(cd, parse(Float64, parts[3]))
        end

    end

    return alpha*pi/180, cl, cd

end


"""
    af_from_files(filenames; Re=[], Mach=[])

Read airfoil file(s) and return a function of the form `cl, cd = func(alpha, Re, M)`

If filenames is just one file, then Re and Mach are ignored (just aoa variation).
af_file_files("somefile.dat")

If filenames is a vector then there is variation with either Re or Mach (both not both).
af_file_files(["f1.dat", "f2.dat", "f3.dat"], Mach=[0.5, 0.6, 0.7])

Filenames can be a matrix for variation with both.
af_file_files(filematrix, Re=[3e6, 5e6], Mach=[0.5, 0.6, 0.7])
where filematrix[i, j] correspnds to Re[i], Mach[j]

Uses the af_from_data function.
"""
function af_from_files(filenames; Re=[], Mach=[])

    # evalutae first file to determine size
    sz = size(filenames)
    nRe = length(Re)
    nMach = length(Mach)

    if length(sz) == 0  # just one file
        alpha, cl, cd = parse_af_file(filenames)

    elseif length(sz) == 1  # list of files
        alpha, cl1, cd1 = parse_af_file(filenames[1])
        nalpha = length(alpha)
        ncond = length(filenames)
    
        cl = zeros(nalpha, ncond)
        cd = zeros(nalpha, ncond)
        cl[:, 1] = cl1
        cd[:, 1] = cd1

        # iterate over remaining files
        for i = 2:ncond
            _, cli, cdi = parse_af_file(filenames[i])
            cl[:, i] = cli
            cd[:, i] = cdi
        end

    else  # matrix of filenames
        alpha, cl1, cd1 = parse_af_file(filenames[1, 1])
        nalpha = length(alpha)
        
        cl = zeros(nalpha, nRe, nMach)
        cd = zeros(nalpha, nRe, nMach)

        for j = 1:nMach
            for i = 1:nRe
                _, clij, cdij = parse_af_file(filenames[i, j])
                cl[:, i, j] = clij
                cd[:, i, j] = cdij
            end
        end

    end

    return af_from_data(alpha, Re, Mach, cl, cd)        
end


"""private function
1d interpolation vs alpha
"""
function afalpha(alpha, Re, Mach, cl, cd)

    nRe = length(Re)
    nMach = length(Mach)

    # squeeze out singleton dimensions if necessary
    if nRe == 1 && nMach == 1  # could be cases with 1, 0, but this would be inconcistent input and I'm not going to bother handlingn it.
        cl = cl[:, 1, 1]
        cd = cd[:, 1, 1]
    elseif nRe == 0 && nMach == 0
        cl = cl[:, 1]
        cd = cd[:, 1]
    end

    afcl = FLOWMath.akima_setup(alpha, cl)
    afcd = FLOWMath.akima_setup(alpha, cd)

    afeval(alpha_pt, Re_pt, M_pt) = FLOWMath.akima_interp(alpha_pt, afcl), 
                                    FLOWMath.akima_interp(alpha_pt, afcd)
    return afeval
end

"""private function
2d interpolation vs alpha, Re
"""
function afalphaRe(alpha, Re, Mach, cl, cd)

    nRe = length(Re)
    nMach = length(Mach)

    # squeeze out singleton dimensions if necessary
    if nMach == 1
        cl = cl[:, :, 1]
        cd = cd[:, :, 1]
    end

    afeval(alpha_pt, Re_pt, M_pt) = FLOWMath.interp2d(FLOWMath.akima, alpha, Re, cl, alpha_pt, Re_pt), 
                                    FLOWMath.interp2d(FLOWMath.akima, alpha, Re, cd, alpha_pt, Re_pt)
    return afeval

end

"""private function
2d interpolation vs alpha, Mach
"""
function afalphaMach(alpha, Re, Mach, cl, cd)

    nRe = length(Re)
    nMach = length(Mach)
    
    # squeeze out singleton dimensions if necessary
    if nRe == 1
        cl = cl[:, 1, :]
        cd = cd[:, 1, :]
    end

    afeval(alpha_pt, Re_pt, M_pt) = FLOWMath.interp2d(FLOWMath.akima, alpha, Mach, cl, alpha_pt, Mach_pt), 
                                    FLOWMath.interp2d(FLOWMath.akima, alpha, Mach, cd, alpha_pt, Mach_pt)
    return afeval
end


"""private function
3d interpolation vs alpha, Re, Mach
"""
function afalphaReMach(alpha, Re, Mach, cl, cd)

    nRe = length(Re)
    nMach = length(Mach)
    
    afeval(alpha_pt, Re_pt, M_pt) = FLOWMath.interp3d(FLOWMath.akima, alpha, Re, Mach, cl, alpha_pt, Re_pt, Mach_pt), 
                                    FLOWMath.interp3d(FLOWMath.akima, alpha, Re, Mach, cd, alpha_pt, Re_pt, Mach_pt)
    return afeval
end


"""
Create an airfoil function directly from alpha, cl, and cd arrays.
The function of the form `cl, cd = func(alpha, Re, M)`
alpha should be in radians.  Uses an akima spline.  af_from_files calls this function.

cl[i, j, k] corresponds to alpha[i], Re[j], Mach[k]

If Mach=[]
cl[i, j] corresponds to alpha[i], Re[j]
size(cl) = (length(alpha), length(Re))
But you can use a singleton dimension for the constant Mach if desired.
size(cl) = (length(alpha), length(Re), 1)
The above also applies for Re=[] where variation is with alpha and Mach.

There is also a convenience method for vector data with just aoa variation
af_from_data(alpha, cl, cd) which just corresponds to af_from_data(alpha, Re=[], Mach=[], cl, cd)
"""
function af_from_data(alpha, Re, Mach, cl, cd)

    nRe = length(Re)
    nMach = length(Mach)

    if nRe <= 1 && nMach <= 1
        afalpha(alpha, Re, Mach, cl, cd)
    elseif nMach <= 1
        afalphaRe(alpha, Re, Mach, cl, cd)
    elseif nRe <= 1
        afalphaMach(alpha, Re, Mach, cl, cd)
    else
        afalphaReMach(alpha, Re, Mach, cl, cd)
    end
end

# convenience wrappers
af_from_data(alpha, cl, cd) = af_from_data(alpha, Re=[], Mach=[], cl, cd)


# ------------ BEM core ------------------


"""
(private) residual function
"""
function residual(phi, rotor, section, op)

    # unpack inputs
    r = section.r
    chord = section.chord
    theta = section.theta
    af = section.af
    Rhub = rotor.Rhub
    Rtip = rotor.Rtip
    B = rotor.B
    turbine = rotor.turbine
    pitch = rotor.pitch
    Vx = op.Vx
    Vy = op.Vy
    rho = op.rho
    mu = op.mu
    asound = op.asound

    # check if turbine or propeller and change input sign if necessary
    setsign = turbine ? 1 : -1
    theta *= setsign
    Vx *= setsign

    # constants
    sigma_p = B*chord/(2.0*pi*r)
    sphi = sin(phi)
    cphi = cos(phi)

    # angle of attack
    alpha = phi - theta

    # Reynolds number
    W0 = sqrt(Vx^2 + Vy^2)  # ignoring induction, which is generally a very minor difference and only affects Reynolds/Mach number
    Re = rho * W0 * chord / mu

    # Mach number
    Mach = W0/asound  # also ignoring induction

    # airfoil cl/cd
    cl_, cd_ = af(alpha, Re, Mach)
    cl, cd = cl_[1], cd_[1]

    # resolve into normal and tangential forces
    cn = cl*cphi + cd*sphi
    ct = cl*sphi - cd*cphi

    # Prandtl's tip and hub loss factor
    factortip = B/2.0*(Rtip - r)/(r*abs(sphi))
    Ftip = 2.0/pi*acos(exp(-factortip))
    factorhub = B/2.0*(r - Rhub)/(Rhub*abs(sphi))
    Fhub = 2.0/pi*acos(exp(-factorhub))
    F = Ftip * Fhub

    # sec parameters
    k = cn*sigma_p/(4.0*F*sphi*sphi)
    kp = ct*sigma_p/(4.0*F*sphi*cphi)

    # parameters used in Vx=0 and Vy=0 cases
    k0 = cn*sigma_p/(4.0*F*sphi*cphi)
    k0p = ct*sigma_p/(4.0*F*sphi*sphi)


    # --- solve for induced velocities ------
    if isapprox(Vx, 0.0, atol=1e-6)

        u = sign(phi)*k0*Vy
        v = zero(phi)
        a = zero(phi)
        ap = zero(phi)

    elseif isapprox(Vy, 0.0, atol=1e-6)
        
        # u = 0.0
        # v = k0p*abs(Vx)
        # a = 0.0
        # ap = 0.0

        u = zero(phi)
        v = k0p*abs(Vx)
        a = zero(phi)
        ap = zero(phi)
    
    else

        if phi < 0
            k = -k
        end

        # if isapprox(k, -1.0, atol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
        #     return 1.0, Outputs()
        # end

        if k <= 2.0/3  # momentum region
            a = k/(1 + k)

        else  # empirical region
            g1 = 2.0*F*k - (10.0/9-F)
            g2 = 2.0*F*k - (4.0/3-F)*F
            g3 = 2.0*F*k - (25.0/9-2*F)

            if isapprox(g3, 0.0, atol=1e-6)  # avoid singularity
                a = 1.0 - 1.0/(2.0*sqrt(g2))
            else
                a = (g1 - sqrt(g2)) / g3
            end
        end

        u = a * Vx

        # -------- tangential induction ----------
        if Vx < 0
            kp = -kp
        end

        # if isapprox(kp, 1.0, atol=1e-6)  # state corresopnds to Vy=0, return any nonzero residual
        #     return 1.0, Outputs()
        # end

        ap = kp/(1 - kp)
        v = ap * Vy

    end


    # ------- residual function -------------
    # R = sin(phi)*(Vy + v) - cos(phi)*(Vx - u)
    # R = sin(phi)/(1 - a) - Vx/Vy*cos(phi)/(1 + ap)
    R = sin(phi)/(Vx - u) - cos(phi)/(Vy + v)

    # ------- loads ---------
    W = sqrt((Vx - u)^2 + (Vy + v)^2)
    Np = cn*0.5*rho*W^2*chord
    Tp = ct*0.5*rho*W^2*chord

    # ---- swap output signs as needed -----
    Tp *= setsign
    v *= setsign

    # The BEM methodology applies hub/tip losses to the loads rather than to the velocities.  
    # This is the most common way to implement a BEM, but it means that the raw velocities are misleading 
    # as they do not contain any hub/tip loss corrections.
    # To fix this we compute the effective hub/tip losses that would produce the same thrust/torque.
    # In other words:
    # CT = 4 a (1 - a) F = 4 a G (1 - a G)\n
    # This is solved for G, then multiplied against the wake velocities.
    G = (1.0 - sqrt(1.0 - 4*a*(1.0 - a)*F))/(2*a)
    u *= G
    v *= G

    return R, Outputs(Np, Tp, a, ap, u, v, phi, alpha, W, cl, cd, cn, ct, F, G)

end


"""
(private) Find a bracket for the root closest to xmin by subdividing
interval (xmin, xmax) into n intervals.

Returns found, xl, xu.
If found = true a bracket was found between (xl, xu)
"""
function firstbracket(f, xmin, xmax, n, backwardsearch=false)

    xvec = range(xmin, xmax, length=n)
    if backwardsearch  # start from xmax and work backwards
        xvec = reverse(xvec)
    end

    fprev = f(xvec[1])
    for i = 2:n
        fnext = f(xvec[i])
        if fprev*fnext < 0  # bracket found
            if backwardsearch
                return true, xvec[i], xvec[i-1]
            else
                return true, xvec[i-1], xvec[i]
            end
        end
        fprev = fnext
    end

    return false, 0.0, 0.0

end


"""
    solve(rotor, section, op)

Solve the BEM equations for given rotor geometry and operating point.

**Arguments**
- `rotor::Rotor`: rotor properties
- `section::Section`: section properties
- `op::OperatingPoint`: operating point

**Returns**
- `outputs::Outputs`: BEM output data including loads, induction factors, etc.
"""
function solve(rotor, section, op)

    # error handling
    if typeof(section) <: Vector
        error("You passed in an vector for section, but this funciton does not accept an vector.\nProbably you intended to use broadcasting (notice the dot): solve.(Ref(rotor), sections, ops)")
    end

    # parameters
    npts = 20  # number of discretization points to find bracket in residual solve

    # unpack
    setsign = rotor.turbine ? 1 : -1
    Vx = op.Vx * setsign  # TODO: ideally should do sign swap only in one place
    Vy = op.Vy
    # theta = section.theta * setsign
    # pitch = op.pitch * setsign

    # ---- determine quadrants based on case -----
    # Vx_is_zero = isapprox(Vx, 0.0, atol=1e-6)
    # Vy_is_zero = isapprox(Vy, 0.0, atol=1e-6)

    # quadrants
    epsilon = 1e-6
    q1 = [epsilon, pi/2]
    q2 = [-pi/2, -epsilon]
    q3 = [pi/2, pi-epsilon]
    q4 = [-pi+epsilon, -pi/2]

    # if Vx_is_zero && Vy_is_zero
    #     return Outputs()

    # elseif Vx_is_zero

    #     startfrom90 = false  # start bracket search from 90 deg instead of 0 deg.

    #     if Vy > 0 && theta > 0
    #         order = (q1, q2)
    #     elseif Vy > 0 && theta < 0
    #         order = (q2, q1)
    #     elseif Vy < 0 && theta > 0
    #         order = (q3, q4)
    #     else  # Vy < 0 && theta < 0
    #         order = (q4, q3)
    #     end

    # elseif Vy_is_zero

    # else  # normal case
    

    # for i = 1:nr

    startfrom90 = false

    if Vx > 0 && Vy > 0
        order = (q1, q2, q3, q4)
    elseif Vx < 0 && Vy > 0
        order = (q2, q1, q4, q3)
    elseif Vx > 0 && Vy < 0
        order = (q3, q4, q1, q2)
    else  # Vx[i] < 0 && Vy[i] < 0
        order = (q4, q3, q2, q1)
    end

        

    # ----- solve residual function ------

    

    # # wrapper to residual function to accomodate format required by fzero
    R(phi) = residual(phi, rotor, section, op)[1]

    success = false
    for j = 1:length(order)  # quadrant orders.  In most cases it should find root in first quadrant searched.
        phimin, phimax = order[j]

        # check to see if it would be faster to reverse the bracket search direction
        backwardsearch = false
        # if !startfrom90
        if phimin == -pi/2 || phimax == -pi/2  # q2 or q4
            backwardsearch = true
        end
        # else
        #     if phimax == pi/2  # q1
        #         backwardsearch = true
        #     end
        # end
        
        # force to dual numbers if necessary
        phimin = phimin*one(section.chord)
        phimax = phimax*one(section.chord)

        # find bracket
        success, phiL, phiU = firstbracket(R, phimin, phimax, npts, backwardsearch)

        # once bracket is found, solve root finding problem and compute loads
        if success

            # phistar = Roots.fzero(R, phiL, phiU)
            
            # so = PyCall.pyimport("scipy.optimize")
            # phistar = so.brentq(R, phiL, phiU)
            phistar = FLOWMath.brent(R, phiL, phiU)
            _, outputs = residual(phistar, rotor, section, op)
            return outputs
        end    
    end    

    # it shouldn't get to this point.  if it does it means no solution was found
    # it will return empty outputs
    # alternatively, one could increase npts and try again
    
    @warn "no solution found for this section"
    return Outputs()
end


# ------------ inflow ------------------



"""
    simpleinflow(Vinf, Omega, r, rho, mu=1.0, asound=1.0, precone=0.0)

Uniform inflow through rotor.  Returns an Inflow object.

**Arguments**
- `Vinf::Float64`: freestream speed (m/s)
- `Omega::Float64`: rotation speed (rad/s)
- `r::Float64`: radial location where inflow is computed (m)
- `precone::Float64`: precone angle (rad)
- `rho::Float64`: air density (kg/m^3)
"""
function simple_op(Vinf, Omega, r, rho, mu=one(rho), asound=one(rho), precone=zero(Vinf))

    # error handling
    if typeof(r) <: Vector
        error("You passed in an vector for r, but this function does not accept an vector.\nProbably you intended to use broadcasting")
    end

    Vx = Vinf * cos(precone) 
    Vy = Omega * r * cos(precone)

    return OperatingPoint(Vx, Vy, rho, mu, asound)

end


"""
    windturbineinflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

Compute relative wind velocity components along blade accounting for inflow conditions
and orientation of turbine.  See Documentation for angle definitions.

**Arguments**
- `Vhub::Float64`: freestream speed at hub (m/s)
- `Omega::Float64`: rotation speed (rad/s)
- `pitch::Float64`: pitch angle (rad)
- `r::Float64`: radial location where inflow is computed (m)
- `precone::Float64`: precone angle (rad)
- `yaw::Float64`: yaw angle (rad)
- `tilt::Float64`: tilt angle (rad)
- `azimuth::Float64`: azimuth angle (rad)
- `hubHt::Float64`: hub height (m) - used for shear
- `shearExp::Float64`: power law shear exponent
- `rho::Float64`: air density (kg/m^3)
- `mu::Float64`: air viscosity (Pa * s)
- `asounnd::Float64`: air speed of sound (m/s)
"""
function windturbine_op(Vhub, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho, mu=1.0, asound=1.0)

    sy = sin(yaw)
    cy = cos(yaw)
    st = sin(tilt)
    ct = cos(tilt)
    sa = sin(azimuth)
    ca = cos(azimuth)
    sc = sin(precone)
    cc = cos(precone)

    # coordinate in azimuthal coordinate system
    x_az = -r*sin(precone)
    z_az = r*cos(precone)
    y_az = 0.0  # could omit (the more general case allows for presweep so this is nonzero)

    # get section heights in wind-aligned coordinate system
    heightFromHub = (y_az*sa + z_az*ca)*ct - x_az*st

    # velocity with shear
    V = Vhub*(1 + heightFromHub/hubHt)^shearExp

    # transform wind to blade c.s.
    Vwind_x = V * ((cy*st*ca + sy*sa)*sc + cy*ct*cc)
    Vwind_y = V * (cy*st*sa - sy*ca)

    # wind from rotation to blade c.s.
    Vrot_x = -Omega*y_az*sc
    Vrot_y = Omega*z_az

    # total velocity
    Vx = Vwind_x + Vrot_x
    Vy = Vwind_y + Vrot_y

    # operating point
    return OperatingPoint(Vx, Vy, rho, mu, asound)

end


function windturbineinflow_az(Vhub, Omega, r, precone, yaw, tilt, azimuth_array, hubHt, shearExp, rho, mu=1.0, asound=1.0)

    azinflows = Array{Array{Inflow}}(undef, 4)
    for i = 1:4
        azinflows[i] = windturbineinflow.(Vhub, Omega, r, precone, yaw, tilt, azimuth_array[i], hubHt, shearExp, rho)
    end

    return azinflows
end


# -------------------------------------


# -------- convenience methods ------------

"""
    thrusttorque(rotor, sections, outputs::Vector{Outputs{TF}}) where TF

integrate the thrust/torque across the blade, 
including 0 loads at hub/tip, using a trapezoidal rule.

**Arguments**
- `rotor::Rotor`: rotor object
- `sections::Vector{Section}`: rotor object
- `outputs::Vector{Outputs}`: output data along blade

**Returns**
- `T::Float64`: thrust (along x-dir see Documentation).
- `Q::Float64`: torque (along x-dir see Documentation).
"""
# function thrusttorque(rotor, sections, outputs)
function thrusttorque(rotor, sections, outputs::Vector{Outputs{TF}}) where TF

    # add hub/tip for complete integration.  loads go to zero at hub/tip.
    rfull = [rotor.Rhub; sections.r; rotor.Rtip]
    Npfull = [0.0; outputs.Np; 0.0]
    Tpfull = [0.0; outputs.Tp; 0.0]

    # integrate Thrust and Torque (trapezoidal)
    thrust = Npfull*cos(rotor.precone)
    torque = Tpfull.*rfull*cos(rotor.precone)

    T = rotor.B * FLOWMath.trapz(rfull, thrust)
    Q = rotor.B * FLOWMath.trapz(rfull, torque)

    return T, Q
end


"""
    thrusttorque(rotor, sections, outputs::Array{Outputs{TF}, 2}) where TF

Integrate the thrust/torque across the blade given an array of output data.
Generally used for azimuthal averaging of thrust/torque.
outputs[i, j] corresponds to sections[i], azimuth[j].  Integrates across azimuth
"""
function thrusttorque(rotor, sections, outputs::Matrix{Outputs{TF}}) where TF

    T = 0.0
    Q = 0.0
    nr, naz = size(outputs)

    for j = 1:naz
        Tsub, Qsub = thrusttorque(rotor, sections, outputs[:, j])
        T += Tsub / n
        Q += Qsub / n
    end

    return T, Q
end




"""
    nondim(T, Q, Vhub, Omega, rho, rotor)

Nondimensionalize the outputs.

**Arguments**
- `T::Float64`: thrust (N)
- `Q::Float64`: torque (N-m)
- `Vhub::Float64`: hub speed used in turbine normalization (m/s)
- `Omega::Float64`: rotation speed used in propeller normalization (rad/s)
- `rho::Float64`: air density (kg/m^3)
- `rotor::Rotor`: rotor object

**Returns**

if windturbine
- `CP::Float64`: power coefficient
- `CT::Float64`: thrust coefficient
- `CQ::Float64`: torque coefficient

if propeller
- `eff::Float64`: efficiency
- `CT::Float64`: thrust coefficient
- `CQ::Float64`: torque coefficient
"""
function nondim(T, Q, Vhub, Omega, rho, rotor)

    P = Q * Omega
    Rp = rotor.Rtip*cos(rotor.precone)

    if rotor.turbine  # wind turbine normalizations

        q = 0.5 * rho * Vhub^2
        A = pi * Rp^2

        CP = P / (q * A * Vhub)
        CT = T / (q * A)
        CQ = Q / (q * Rp * A)

        return CP, CT, CQ

    else  # propeller

        n = Omega/(2*pi)
        Dp = 2*Rp

        if T < 0
            eff = 0.0  # creating drag not thrust
        else
            eff = T*Vhub/P
        end
        CT = T / (rho * n^2 * Dp^4)
        CQ = Q / (rho * n^2 * Dp^5)

        return eff, CT, CQ

    # elseif rotortype == "helicopter"

    #     A = pi * Rp^2

    #     CT = T / (rho * A * (Omega*Rp)^2)
    #     CP = P / (rho * A * (Omega*Rp)^3)
    #     FM = CT^(3/2)/(sqrt(2)*CP)

    #     return FM, CT, CP
    end

end


end  # module
