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


import Dierckx  # cubic b-spline for airfoil data
import Roots  # solve residual equation
import Parameters: @unpack


export Section, Rotor, Inflow, Outputs
export af_from_file, af_from_data
export simpleinflow, windturbineinflow, windturbineinflowmultiple
export solve, loads, effectivewake, thrusttorque, nondim



# --------- structs -------------


"""
    Rotor(r, chord, theta, af, Rhub, Rtip, B, precone)

Define rotor geometry.

**Arguments**
- `r::Array{Float64, 1}`: radial locations (m)
- `chord::Array{Float64, 1}`: chord lengths (m)
- `theta::Array{Float64, 1}`: total twist including pitch (rad)
- `af::Array{AirfoilData, 1}`: airfoils
- `Rhub::Float64`: hub radius (along blade length)
- `Rtip::Float64`: tip radius (along blade length)
- `B::Int64`: number of blades
- `precone::Float64`: precone angle (rad)
"""
# struct Rotor{TF, TI, TAF}
#     r::Array{TF, 1}
#     chord::Array{TF, 1}
#     theta::Array{TF, 1}
#     af::Array{TAF, 1}
#     Rhub::TF
#     Rtip::TF
#     B::TI
#     precone::TF
# end


struct Section{TF, TAF}

    r::TF
    chord::TF
    theta::TF  # includes pitch
    af::TAF

end

struct Rotor{TF, TI, TB}
    
    Rhub::TF
    Rtip::TF
    B::TI
    turbine::TB

end

# make rotor broadcastable as a single entity
Base.Broadcast.broadcastable(r::Rotor) = Ref(r) 

# operating point for the turbine/propeller
struct Inflow{TF}
    Vx::TF
    Vy::TF
    rho::TF
    mu::TF
    asound::TF
end

Inflow(Vx, Vy, rho) = Inflow(Vx, Vy, rho, 1.0, 1.0)  # if Re and M are unnecessary


struct Outputs{TF}

    Np::TF
    Tp::TF
    a::TF
    ap::TF
    u::TF
    v::TF
    phi::TF
    W::TF
    cl::TF
    cd::TF
    F::TF

end

# constructor for case with no solution found
Outputs() = Outputs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# -------------------------------



# ----------- airfoil ---------------


"""
    af_from_file(filename)

Read an airfoil file.
Currently only reads one Reynolds number.
Additional data like cm is optional but will be ignored.
alpha should be in degrees

format:

header
alpha1 cl1 cd1
alpha2 cl2 cd2
alpha3 cl3 cd3
...

Returns an AirfoilData object
"""
function af_from_file(filename)

    alpha = Float64[]
    cl = Float64[]
    cd = Float64[]

    open(filename) do f

        # skip header
        readline(f)

        for line in eachline(f)
            parts = split(line)
            push!(alpha, parse(Float64, parts[1]))
            push!(cl, parse(Float64, parts[2]))
            push!(cd, parse(Float64, parts[3]))
        end

    end

    return af_from_data(alpha*pi/180.0, cl, cd)
end



"""
    af_from_data(alpha, cl, cd)

Create an AirfoilData object directly from alpha, cl, and cd arrays.
alpha should be in radians.

af_from_file calls this function indirectly.  Uses a cubic B-spline
(if the order of the data permits it).  A small amount of smoothing of
lift and drag coefficients is also applied to aid performance
for gradient-based optimization.
"""
function af_from_data(alpha, cl, cd)

    # # TODO: update once smoothing is implemented: https://github.com/JuliaMath/Interpolations.jl/issues/254
    # afcl = CubicSplineInterpolation(alpha, cl)
    # afcd = CubicSplineInterpolation(alpha, cd)
    # af = AirfoilData(afcl, afcd)

    k = min(length(alpha)-1, 3)  # can't use cubic spline is number of entries in alpha is small

    # 1D interpolations for now.  ignoring Re dependence (which is very minor)
    afcl_1d = Dierckx.Spline1D(alpha, cl; k=k, s=0.1)
    afcd_1d = Dierckx.Spline1D(alpha, cd; k=k, s=0.001)

    afeval(alpha, Re, M) = afcl_1d(alpha), afcd_1d(alpha)  # ignore Re, M 

    return afeval
end


# """
# private

# evalute airfoil spline at alpha (rad), Reynolds number, Mach number
# """
# function airfoil(af, alpha, Re, M)

#     cl, cd = af(alpha, Re, M)

#     return cl, cd
# end


# function airfoil(af::AirfoilData, alpha::T) where {T<:ForwardDiff.Dual}

#     a = Gradients.values(alpha)[1]
#     cl, cd = airfoil(af, a)

#     dclda = Dierckx.derivative(af.cl, a)
#     dcdda = Dierckx.derivative(af.cd, a)

#     cldual = Gradients.manualderiv(cl, alpha, dclda)
#     cddual = Gradients.manualderiv(cd, alpha, dcdda)

#     return cldual, cddual
# end



# ---------------------------------






# ------------ BEM core ------------------



"""
(private)
residual function
base calculations used in normal case, Vx=0 case, and Vy=0 case
"""
function residual(phi, section, inflow, rotor)

    # unpack inputs
    @unpack r, chord, theta, af = section
    @unpack Vx, Vy, rho, mu, asound = inflow
    @unpack Rhub, Rtip, B, turbine = rotor

    # check if turbine or propeller and change input sign if necessary
    swapsign = turbine ? 1 : -1
    theta *= swapsign
    Vx *= swapsign
    # Vy *= swapsign

    # constants
    sigma_p = B*chord/(2.0*pi*r)
    sphi = sin(phi)
    cphi = cos(phi)

    # angle of attack
    alpha = phi - theta

    # Reynolds number
    W0 = sqrt(Vx^2 + Vy^2)  # ignoring induction, which is generally a very minor error and only affects Reynolds number
    Re = rho * W0 * chord / mu

    # Mach number
    Mach = W0/asound  # also ignoring induction

    # airfoil cl/cd
    cl, cd = af(alpha, Re, Mach)

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
        v = 0.0
        a = 0.0
        ap = 0.0

    elseif isapprox(Vy, 0.0, atol=1e-6)
        
        u = 0.0
        v = k0p*abs(Vx)
        a = 0.0
        ap = 0.0
    
    else

        if phi < 0
            k = -k
        end

        if isapprox(k, -1.0, atol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
            return 1.0, Outputs()
        end

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

        if isapprox(kp, 1.0, atol=1e-6)  # state corresopnds to Vy=0, return any nonzero residual
            return 1.0, Outputs()
        end

        ap = kp/(1 - kp)
        v = ap * Vy
    end


    # ------- residual function -------------
    # R = sin(phi)*(Vy + v) - cos(phi)*(Vx - u)
    # R = sin(phi)/(1 - a) - Vx/Vy*cos(phi)/(1 + ap)
    R = sin(phi)/(Vx - u) - cos(phi)/(Vy + v)

    # ------- loads ---------
    W2 = (Vx - u)^2 + (Vy + v)^2
    Np = cn*0.5*rho*W2*chord
    Tp = ct*0.5*rho*W2*chord

    # ---- swap output signs as needed -----
    Tp *= swapsign
    v *= swapsign

    return R, Outputs(Np, Tp, a, ap, u, v, phi, sqrt(W2), cl, cd, F)  # multiply by F because a and ap as currently used are only correct in combination with the loads.  If you want a wake model then you need to add the hub/tip loss factors separately.

end


# """
# (private)
# residual for normal case
# """
# function residual(phi, section, inflow, rotor)

#     k, kp, F, _, _, Nu, Tu = residualbase(phi, section, inflow, rotor)

#     Vx = inflow.Vx
#     Vy = inflow.Vy

#     # -------- axial induction ----------
#     if phi < 0
#         k = -k
#     end

#     if isapprox(k, -1.0, atol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
#         return 1.0, Outputs()
#     end

#     if k <= 2.0/3  # momentum region
#         a = k/(1 + k)

#     else  # empirical region
#         g1 = 2.0*F*k - (10.0/9-F)
#         g2 = 2.0*F*k - (4.0/3-F)*F
#         g3 = 2.0*F*k - (25.0/9-2*F)

#         if isapprox(g3, 0.0, atol=1e-6)  # avoid singularity
#             a = 1.0 - 1.0/(2.0*sqrt(g2))
#         else
#             a = (g1 - sqrt(g2)) / g3
#         end
#     end

#     # -------- tangential induction ----------
#     if Vx < 0
#         kp = -kp
#     end

#     if isapprox(kp, 1.0, atol=1e-6)  # state corresopnds to Vy=0, return any nonzero residual
#         R = 1.0
#         return 1.0, Outputs()
#     end

#     ap = kp/(1 - kp)


#     # ------- residual function -------------
#     R = sin(phi)/(1 - a) - Vx/Vy*cos(phi)/(1 + ap)

#     # ------- loads ---------
#     W2 = (Vx*(1-a))^2 + (Vy*(1+ap))^2
#     Np = Nu*W2
#     Tp = Tu*W2

#     return R, Outputs(Np, Tp, a, ap, phi, sqrt(W), Vx*a*F, Vy*ap*F, F)  # multiply by F because a and ap as currently used are only correct in combination with the loads.  If you want a wake model then you need to add the hub/tip loss factors separately.
# end



# """
# (private)
# residual for Vx=0 case
# """
# function residualVx0(phi, section, inflow, rotor)

#     # base params
#     _, _, F, k2, _, Nu, Tu = residualbase(phi, section, inflow, rotor)

#     Vy = inflow.Vy
#     os = -sign(phi)

#     if os*k2 < 0  # return any nonzero residual
#         return 1.0, Outputs()
#     end


#     # induced velocity
#     u = os*sqrt(os*k2)*abs(Vy)

#     # residual
#     R = - sin(phi)/u - cos(phi)/Vy

#     # loads
#     W2 = u^2 + Vy^2
#     Np = Nu*W2
#     Tp = Tu*W2

#     return R, Outputs(Np, Tp, 0.0, 0.0, phi, sqrt(W2), u, 0.0, F)  # F already included in these velocity deficits.  a, ap both meaningless

# end



# """
# (private)
# residual for Vy=0 case
# """
# function residualVy0(phi, section, inflow, rotor)

#     # base params
#     _, _, F, _, kp2, Nu, Tu = residualbase(phi, section, inflow, rotor)

#     Vx = inflow.Vx
    
#     # induced velocity
#     v = kp2*abs(Vx)

#     # residual
#     R = v*sin(phi) - Vx*cos(phi)

#     # loads
#     W2 = v^2 + Vx^2
#     Np = Nu*W2
#     Tp = Tu*W2

#     return R, Outputs(Np, Tp, 0.0, 0.0, phi, sqrt(W2), 0.0, v, F)
# end



"""
(private)
Find a bracket for the root closest to xmin by subdividing
interval (xmin, xmax) into n intervals

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
distributed loads for one section
"""
function solve(section, inflow, rotor)

    # error handling
    if typeof(section) <: Array
        error("You passed in an array, but this funciton does not accept an array.\nProbably you intended to use broadcasting (notice the dot): solve.(sections, inflows, rotor)")
    end

    # parameters
    npts = 20  # number of discretization points to find bracket in residual solve

    # unpack
    swapsign = rotor.turbine ? 1 : -1
    Vx = inflow.Vx * swapsign  # TODO: shouldn't be doing the sign swap in two different places
    Vy = inflow.Vy
    twist = section.theta * swapsign

    # ---- determine quadrants based on case -----
    Vx_is_zero = isapprox(Vx, 0.0, atol=1e-6)
    Vy_is_zero = isapprox(Vy, 0.0, atol=1e-6)

    # quadrants
    epsilon = 1e-6
    q1 = [epsilon, pi/2]
    q2 = [-pi/2, -epsilon]
    q3 = [pi/2, pi-epsilon]
    q4 = [-pi+epsilon, -pi/2]

    if Vx_is_zero && Vy_is_zero
        return Outputs()

    elseif Vx_is_zero

        startfrom90 = false  # start bracket search from 90 deg instead of 0 deg.

        if Vy > 0 && twist > 0
            order = (q1, q2)
        elseif Vy > 0 && twist < 0
            order = (q2, q1)
        elseif Vy < 0 && twist > 0
            order = (q3, q4)
        else  # Vy < 0 && twist < 0
            order = (q4, q3)
        end

    elseif Vy_is_zero

        startfrom90 = true  # start bracket search from 90 deg

        if Vx > 0 && abs(twist) < pi/2
            order = (q1, q3)
        elseif Vx < 0 && abs(twist) < pi/2
            order = (q2, q4)
        elseif Vx > 0 && abs(twist) > pi/2
            order = (q3, q1)
        else  # Vx < 0 && abs(twist) > pi/2
            order = (q4, q2)
        end

    else  # normal case

        startfrom90 = false

        if Vx > 0 && Vy > 0
            order = (q1, q2, q3, q4)
        elseif Vx < 0 && Vy > 0
            order = (q2, q1, q4, q3)
        elseif Vx > 0 && Vy < 0
            order = (q3, q4, q1, q2)
        else  # Vx < 0 && Vy < 0
            order = (q4, q3, q2, q1)
        end

    end

    # ----- solve residual function ------

    # wrapper to residual function to accomodate format required by fzero
    function R(phi)
        zero, _ = residual(phi, section, inflow, rotor)
        return zero
    end

    success = false
    for j = 1:length(order)  # quadrant orders.  In most cases it should find root in first quadrant searched.
        phimin, phimax = order[j]

        # check to see if it would be faster to reverse the bracket search direction
        backwardsearch = false
        if !startfrom90
            if phimin == -pi/2 || phimax == -pi/2  # q2 or q4
                backwardsearch = true
            end
        else
            if phimax == pi/2  # q1
                backwardsearch = true
            end
        end

        # find bracket
        success, phiL, phiU = firstbracket(R, phimin, phimax, npts, backwardsearch)

        # once bracket is found, solve root finding problem and compute loads
        if success

            phistar = Roots.fzero(R, phiL, phiU)
            _, outputs = residual(phistar, section, inflow, rotor)

            return outputs
        end        
    end

    # it shouldn't get to this point.  if it does it means no solution was found
    # it will return empty outputs
    # alternatively, one could increase npts and try again
    
    return Outputs()
end


"""
    distributedloads(rotor::Rotor, inflow::Inflow, turbine::Bool)

Compute the distributed loads along blade at specified condition.
turbine can be true/false depending on if the analysis is for a turbine or prop
(just affects some input/output conventions as noted in the user doc).

**Returns**
- `Np::Array{Float64, 1}`: force per unit length in the normal direction (N/m)
- `Tp::Array{Float64, 1}`: force per unit length in the tangential direction (N/m)
- `uvec::Array{Float64, 1}`: induced velocity in x direction
- `vvec::Array{Float64, 1}`: induced velocity in y direction
"""



# ------------ inflow ------------------



"""
    simpleinflow(Vinf, Omega, r, precone, rho)

Uniform inflow through rotor.  Returns an Inflow object.

**Arguments**
- `Vinf::Float64`: freestream speed (m/s)
- `Omega::Float64`: rotation speed (rad/s)
- `r::Float64`: radial location where inflow is computed (m)
- `precone::Float64`: precone angle (rad)
- `rho::Float64`: air density (kg/m^3)
"""
function simpleinflow(Vinf, Omega, r, rho, mu=1.0, asound=1.0, precone=0.0)

    Vx = Vinf * cos(precone)
    Vy = Omega * r * cos(precone)

    return Inflow(Vx, Vy, rho, mu, asound)

end

"""
    windturbineinflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

Compute relative wind velocity components along blade accounting for inflow conditions
and orientation of turbine.  See theory doc for angle definitions.

**Arguments**
- `Vinf::Float64`: freestream speed (m/s)
- `Omega::Float64`: rotation speed (rad/s)
- `r::Array{Float64, 1}`: radial locations where inflow is computed (m)
- `precone::Float64`: precone angle (rad)
- `yaw::Float64`: yaw angle (rad)
- `tilt::Float64`: tilt angle (rad)
- `azimuth::Float64`: azimuth angle (rad)
- `hubHt::Float64`: hub height (m) - used for shear
- `shearExp::Float64`: power law shear exponent
- `rho::Float64`: air density (kg/m^3)
"""
function windturbineinflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho, mu, asound)

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
    V = Vinf*(1 + heightFromHub/hubHt)^shearExp

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
    return Inflow(Vx, Vy, rho, mu, asound)

end


# """
#     windturbineinflowmultiple(nsectors, Vinf, Omega, r, precone, yaw, tilt, hubHt, shearExp, rho)

# Convenience function that calls windturbineinflow multiple times, once for each azimuthal angle.
# The azimuth angles are uniformly spaced (starting at 0) based on the number of sectors that
# the user wishes to divide the rotor into.
# """
# function windturbineinflowmultiple(nsectors, Vinf, Omega, r, precone, yaw, tilt, hubHt, shearExp, rho)


#     infs = Array{Inflow}(undef, nsectors)

#     for j = 1:nsectors
#         azimuth = 2*pi*float(j)/nsectors
#         infs[j] = windturbineinflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)
#     end

#     return infs
# end



# -------------------------------------


# -------- convenience methods ------------

function loads(outputs)
    Np = getfield.(outputs, :Np)
    Tp = getfield.(outputs, :Tp)
    
    return Np, Tp
end


function effectivewake(outputs)

    u = getfield.(outputs, :u)
    v = getfield.(outputs, :v)
    F = getfield.(outputs, :F)
    a = getfield.(outputs, :a)
    
    # the effective "F" if it were multiplied against the velocities instead of the forces
    G = (1.0 .- sqrt.(1.0 .- 4*a.*(1.0 .- a).*F))./(2*a)

    ueff = u.*G
    veff = v.*G

    return ueff, veff

end


"""
integrate the thrust/torque across the blade, including 0 loads at hub/tip
"""
function thrusttorque(Rhub, r, Rtip, Np, Tp, B, precone=0.0)

    # add hub/tip for complete integration.  loads go to zero at hub/tip.
    rfull = [Rhub; r; Rtip]
    Npfull = [0.0; Np; 0.0]
    Tpfull = [0.0; Tp; 0.0]

    # integrate Thrust and Torque (trapezoidal)
    thrust = Npfull*cos(precone)
    torque = Tpfull.*rfull*cos(precone)

    T = B * trapz(rfull, thrust)
    Q = B * trapz(rfull, torque)

    return T, Q
end


"""
(private)
trapezoidal integration
"""
function trapz(x, y)  # integrate y w.r.t. x

    integral = 0.0
    for i = 1:length(x)-1
        integral += (x[i+1]-x[i])*0.5*(y[i] + y[i+1])
    end
    return integral
end


"""
thrusttorque(rotor::Rotor, inflow::Array{Inflow, 1}, turbine::Bool)

Compute thrust and toruqe at the provided inflow conditions.

**Returns**
- `T::Float64`: thrust (N)
- `Q::Float64`: torque (N-m)
"""
# function thrusttorque(rotor::Rotor, inflow::Array{Inflow, 1}, turbine::Bool)  #, nsectors::Int64)
# function thrusttorque(sections, inflows, rotor

#     nsectors = length(inflow)  # number of sectors (should be evenly spaced)
#     T = 0.0
#     Q = 0.0

#     # coarse integration - rectangle rule
#     for j = 1:nsectors  # integrate across azimuth

#         Np, Tp = distributedloads(rotor, inflow[j], turbine)

#         Tsub, Qsub = thrusttorqueintegrate(rotor.Rhub, rotor.r, rotor.Rtip, rotor.precone, Np, Tp)

#         T += rotor.B * Tsub / nsectors
#         Q += rotor.B * Qsub / nsectors
#     end

#     return T, Q

# end


"""
Nondimensionalize the outputs.

**Arguments**
- `T::Float64`: thrust (N)
- `Q::Float64`: torque (N-m)
- `Vhub::Float64`: hub speed used in turbine normalization (m/s)
- `Omega::Float64`: rotation speed used in propeller normalization (rad/s)
- `rho::Float64`: air density (kg/m^3)
- `Rtip::Float64`: rotor tip length (m)
- `precone::Float64`: precone angle (rad)
- `turbine::Bool`: turbine (true) or propeller (false)

**Returns**

if turbine
- `CP::Float64`: power coefficient
- `CT::Float64`: thrust coefficient
- `CQ::Float64`: torque coefficient

if propeller
- `eff::Float64`: efficiency
- `CT::Float64`: thrust coefficient
- `CQ::Float64`: torque coefficient
"""
function nondim(T, Q, Vhub, Omega, rho, Rtip, rotortype, precone=0.0)

    P = Q * Omega
    Rp = Rtip*cos(precone)

    if rotortype == "windturbine"

        q = 0.5 * rho * Vhub^2
        A = pi * Rp^2

        CP = P / (q * A * Vhub)
        CT = T / (q * A)
        CQ = Q / (q * Rp * A)

        return CP, CT, CQ

    elseif rotortype == "propeller"

        n = Omega/(2*pi)
        Dp = 2*Rp

        if T < 0
            eff = 0  # creating drag not thrust
        else
            eff = T*Vhub/P
        end
        CT = T / (rho * n^2 * Dp^4)
        CQ = Q / (rho * n^2 * Dp^5)

        return eff, CT, CQ

    elseif rotortype == "helicopter"

        A = pi * Rp^2

        CT = T / (rho * A * (Omega*Rp)^2)
        CP = P / (rho * A * (Omega*Rp)^3)
        FM = CT^(3/2)/(sqrt(2)*CP)

        return FM, CT, CP
    end

end


end  # module
