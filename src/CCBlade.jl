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
import Interpolations
import ForwardDiff
import OpenMDAO


export Section, Rotor, Inflow, Outputs
export af_from_file, af_from_data
export simpleinflow, windturbineinflow, windturbineinflow_az
export solve, loads, effectivewake, thrusttorque, thrusttorque_azavg, nondim

include("openmdao.jl")


# --------- structs -------------


"""
    Section(r, chord, theta, af)

Define section properties at a given radial location on the rotor

**Arguments**
- `r::Float64`: radial location (Rhub < r < Rtip)
- `chord::Float64`: local chord length
- `theta::Float64`: twist angle (radians)
- `af`: a function of the form: cl, cd = af(alpha, Re, Mach)
"""
mutable struct Section{TF, TAF}

    r::TF
    chord::TF
    theta::TF  # includes pitch
    af::TAF

end

"""
    Rotor(Rhub, Rtip, B, turbine, precone=0.0)

Define geometry common to the entire rotor.

**Arguments**
- `Rhub::Float64`: hub radius (along blade length)
- `Rtip::Float64`: tip radius (along blade length)
- `B::Int64`: number of blades
- `turbine::Bool`: true if turbine, false if propeller
- `precone::Float64`: precone angle
"""
mutable struct Rotor{TF, TI, TB}
    
    Rhub::TF
    Rtip::TF
    B::TI
    turbine::TB
    precone::TF

end

# convenience constructor for no precone
Rotor(Rhub, Rtip, B, turbine) = Rotor(Rhub, Rtip, B, turbine, 0.0)

# make rotor broadcastable as a single entity
# Base.Broadcast.broadcastable(r::Rotor) = Ref(r) 


"""
    Inflow(Vx, Vy, rho, mu=1.0, asound=1.0)

Operation point for a rotor.  The x direction is the axial direction, and y direction is the tangential direction in the rotor plane.  See Documentation for more detail on coordinate systems.

**Arguments**
- `Vx::Float64`: velocity in x-direction
- `Vy::Float64`: velocity in y-direction
- `rho::Float64`: fluid density
- `mu::Float64`: fluid dynamic viscosity (unused if Re not included in airfoil data)
- `asound::Float64`: fluid speed of sound (unused if Mach not included in airfoil data)
"""
mutable struct Inflow{TF}
    Vx::TF
    Vy::TF
    rho::TF
    mu::TF
    asound::TF
end

# convenience constructor when Re and Mach are not used.
Inflow(Vx, Vy, rho) = Inflow(Vx, Vy, rho, 1.0, 1.0) 


"""
    Outputs(Np, Tp, a, ap, u, v, phi, W, cl, cd, F)

Outputs from the BEM solver.

**Arguments**
- `Np::Float64`: normal force per unit length
- `Tp::Float64`: tangential force per unit length
- `a::Float64`: axial induction factor
- `ap::Float64`: tangential induction factor
- `u::Float64`: axial induced velocity
- `v::Float64`: tangential induced velocity
- `phi::Float64`: inflow angle
- `W::Float64`: inflow velocity
- `cl::Float64`: lift coefficient
- `cd::Float64`: drag coefficient
- `F::Float64`: hub/tip loss correction
"""
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
Outputs() = Outputs(0.0, 0.0, 0.0, 0.0, 0.1, 0.1, -5.0*pi/180.0, 0.0, 0.0, 0.0, 0.0)

struct PartialsWrt{TF}
    phi::TF
    r::TF
    chord::TF
    theta::TF
    Vx::TF
    Vy::TF
    rho::TF
    mu::TF
    asound::TF
    Rhub::TF
    Rtip::TF
    precone::TF
end

# Need this for the mapslices call in output_partials.
PartialsWrt(x::AbstractArray) = PartialsWrt(x...)
    


# -------------------------------



# ----------- airfoil ---------------


"""
    af_from_file(filename)

Read an airfoil file.
Currently only reads one Reynolds number.
Additional data like cm is optional but will be ignored.
alpha should be in degrees

format:

header\n
alpha1 cl1 cd1\n
alpha2 cl2 cd2\n
alpha3 cl3 cd3\n
...

Returns a function of the form `cl, cd = func(alpha, Re, M)` although Re and M are currently ignored.
"""
function af_from_file(filename; use_interpolations_jl=false)

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

    return af_from_data(alpha*pi/180.0, cl, cd; use_interpolations_jl=use_interpolations_jl)
end



"""
    af_from_data(alpha, cl, cd)

Create an AirfoilData object directly from alpha, cl, and cd arrays.
alpha should be in radians.

`af_from_file` calls this function indirectly.  Uses a cubic B-spline
(if the order of the data permits it).  A small amount of smoothing of
lift and drag coefficients is also applied to aid performance
for gradient-based optimization.

Returns a function of the form `cl, cd = func(alpha, Re, M)` although Re and M are currently ignored.
"""
function af_from_data(alpha, cl, cd, spl_k=3; use_interpolations_jl=false)

    # # TODO: update once smoothing is implemented: https://github.com/JuliaMath/Interpolations.jl/issues/254
    # afcl = CubicSplineInterpolation(alpha, cl)
    # afcd = CubicSplineInterpolation(alpha, cd)
    # af = AirfoilData(afcl, afcd)

    k = min(length(alpha)-1, spl_k)  # can't use cubic spline is number of entries in alpha is small

    # 1D interpolations for now.  ignoring Re dependence (which is very minor)
    afcl_1d = Dierckx.Spline1D(alpha, cl; k=k, s=0.1)
    afcd_1d = Dierckx.Spline1D(alpha, cd; k=k, s=0.001)

    if use_interpolations_jl

        # Check if the original alpha is uniformly distributed.
        dalpha = alpha[2:end] - alpha[1:end-1]
        tol = 1e-8
        if all(@. abs(dalpha - dalpha[1]) < tol)
            # Uniform spacing, so no need to reinterpolate the Dierckx
            # interpolation onto a uniform grid.

            # println("detected uniform angle of attack sampling.")
            n_uniform = length(alpha)
            alpha_uniform = LinRange(minimum(alpha), maximum(alpha), n_uniform)
            afcl_1d = Interpolations.CubicSplineInterpolation(alpha_uniform, cl, extrapolation_bc=Interpolations.Periodic())
            afcd_1d = Interpolations.CubicSplineInterpolation(alpha_uniform, cd, extrapolation_bc=Interpolations.Periodic())

        else
            # Evaluate the Dierckx interpolation object on a uniform grid.
            n_uniform = 2*length(alpha)
            alpha_uniform = LinRange(minimum(alpha), maximum(alpha), n_uniform)
            cl_uniform = afcl_1d(alpha_uniform)
            cd_uniform = afcd_1d(alpha_uniform)

            # Use the uniform data to create a new interpolations objects using
            # Interpolations.jl.
            afcl_1d = Interpolations.CubicSplineInterpolation(alpha_uniform, cl_uniform, extrapolation_bc=Interpolations.Periodic())
            afcd_1d = Interpolations.CubicSplineInterpolation(alpha_uniform, cd_uniform, extrapolation_bc=Interpolations.Periodic())
        end

    end

    afeval(alpha, Re, M) = afcl_1d(alpha), afcd_1d(alpha)  # ignore Re, M 

    return afeval
end



# ---------------------------------






# ------------ BEM core ------------------


"""
(private) residual function
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
    # F::typeof(Ftip) = 1.0

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
    W2 = (Vx - u)^2 + (Vy + v)^2
    Np = cn*0.5*rho*W2*chord
    Tp = ct*0.5*rho*W2*chord

    # ---- swap output signs as needed -----
    Tp *= swapsign
    v *= swapsign

    # @show Np, Tp, u, phi
    return R, Outputs(Np, Tp, a, ap, u, v, phi, sqrt(W2), cl, cd, F)  # multiply by F because a and ap as currently used are only correct in combination with the loads.  If you want a wake model then you need to add the hub/tip loss factors separately.

end

function residual_partials(phi, section, inflow, rotor)
    # unpack inputs
    @unpack r, chord, theta, af = section
    @unpack Vx, Vy, rho, mu, asound = inflow
    @unpack Rhub, Rtip, B, turbine, precone = rotor

    # Get a version of the residual function that's compatible with ForwardDiff.
    function R(inputs)
        # The order of inputs should always match the order of fields in the
        # PartialsWrt struct.
        _phi = inputs[1]
        _r = inputs[2]
        _chord = inputs[3]
        _theta = inputs[4]
        _Vx, _Vy, _rho, _mu, _asound = inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]
        _Rhub, _Rtip, _precone = inputs[10], inputs[11], inputs[12]
        _section = Section(_r, _chord, _theta, af)
        _inflow = Inflow(_Vx, _Vy, _rho, _mu, _asound)
        _rotor = Rotor(_Rhub, _Rtip, B, turbine, _precone)
        res, out = residual(_phi, _section, _inflow, _rotor)
        return res
    end

    # Do it.
    x = [phi, r, chord, theta, Vx, Vy, rho, mu, asound, Rhub, Rtip, precone]
    return PartialsWrt(ForwardDiff.gradient(R, x))

end

function output_partials(phi, section, inflow, rotor)
    # unpack inputs
    @unpack r, chord, theta, af = section
    @unpack Vx, Vy, rho, mu, asound = inflow
    @unpack Rhub, Rtip, B, turbine, precone = rotor

    # Get a version of the output function that's compatible with ForwardDiff.
    function R(inputs)
        # The order of inputs should always match the order of fields in the
        # PartialsWrt struct.
        _phi = inputs[1]
        _r = inputs[2]
        _chord = inputs[3]
        _theta = inputs[4]
        _Vx, _Vy, _rho, _mu, _asound = inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]
        _Rhub, _Rtip, _precone = inputs[10], inputs[11], inputs[12]
        _section = Section(_r, _chord, _theta, af)
        _inflow = Inflow(_Vx, _Vy, _rho, _mu, _asound)
        _rotor = Rotor(_Rhub, _Rtip, B, turbine, _precone)
        res, out = residual(_phi, _section, _inflow, _rotor)
        return [getfield(out, i) for i in fieldnames(typeof(out))]
    end

    # Do it.
    x = [phi, r, chord, theta, Vx, Vy, rho, mu, asound, Rhub, Rtip, precone]
    return Outputs(mapslices(PartialsWrt, ForwardDiff.jacobian(R, x), dims=2)...)
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


function firstbracket(rotor::Rotor, section::Section, inflow::Inflow)

    # parameters
    npts = 20  # number of discretization points to find bracket in residual solve

    # unpack
    swapsign = rotor.turbine ? 1 : -1
    Vx = inflow.Vx * swapsign  # TODO: shouldn't be doing the sign swap in two different places
    Vy = inflow.Vy
    theta = section.theta * swapsign

    # ---- determine quadrants based on case -----
    Vx_is_zero = isapprox(Vx, 0.0, atol=1e-6)
    Vy_is_zero = isapprox(Vy, 0.0, atol=1e-6)

    # quadrants
    epsilon = 1e-6
    q1 = [epsilon, pi/2]
    q2 = [-pi/2, -epsilon]
    q3 = [pi/2, pi-epsilon]
    q4 = [-pi+epsilon, -pi/2]

    # wrapper to residual function to accomodate format required by the
    # firstbracket routine above.
    function R(phi)
        zero, _ = residual(phi, section, inflow, rotor)
        return zero
    end

    if Vx_is_zero && Vy_is_zero
        return Outputs()

    elseif Vx_is_zero

        startfrom90 = false  # start bracket search from 90 deg instead of 0 deg.

        if Vy > 0 && theta > 0
            order = (q1, q2)
        elseif Vy > 0 && theta < 0
            order = (q2, q1)
        elseif Vy < 0 && theta > 0
            order = (q3, q4)
        else  # Vy < 0 && theta < 0
            order = (q4, q3)
        end

    elseif Vy_is_zero

        startfrom90 = true  # start bracket search from 90 deg

        if Vx > 0 && abs(theta) < pi/2
            order = (q1, q3)
        elseif Vx < 0 && abs(theta) < pi/2
            order = (q2, q4)
        elseif Vx > 0 && abs(theta) > pi/2
            order = (q3, q1)
        else  # Vx < 0 && abs(theta) > pi/2
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

        # once bracket is found, return it.
        if success
            return success, phiL, phiU
        end        
    end

    # If we get to this point, we've failed to find a bracket.
    return success, zero(rotor.Rhub), zero(rotor.Rhub)
end


"""
    solve(section, inflow, rotor)

Solve the BEM equations for one section, with given inflow conditions, and rotor properties.
If multiple sections are to be solved (typical usage) then one can use broadcasting:
`solve.(sections, inflows, rotor)` where sections and inflows are arrays.

**Arguments**
- `rotor::Rotor`: rotor properties
- `section::Section`: section properties
- `inflow::Inflow`: inflow conditions

**Returns**
- `outputs::Outputs`: BEM output data including loads, induction factors, etc.
"""
function solve(rotor, section, inflow)

    # error handling
    if typeof(section) <: Array
        error("You passed in an array for section, but this funciton does not accept an array.\nProbably you intended to use broadcasting (notice the dot): solve.(sections, inflows, rotor)")
    end

    # ----- solve residual function ------

    # wrapper to residual function to accomodate format required by fzero
    function R(phi)
        zero, _ = residual(phi, section, inflow, rotor)
        return zero
    end

    # Attempt to bracket the solution.
    success, phiL, phiU = firstbracket(rotor, section, inflow)

    if success

        phistar = Roots.fzero(R, phiL, phiU)
        _, outputs = residual(phistar, section, inflow, rotor)

        return success, outputs
    else
        # it shouldn't get to this point.  if it does it means no solution was found
        # it will return empty outputs
        # alternatively, one could increase npts and try again
    
        return false, Outputs()
    end        
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
function simpleinflow(Vinf, Omega, r, rho, mu=1.0, asound=1.0, precone=0.0)

    # error handling
    if typeof(r) <: Array
        error("You passed in an array for r, but this function does not accept an array.\nProbably you intended to use broadcasting")
    end

    Vx = Vinf * cos(precone)
    Vy = Omega * r * cos(precone)

    return Inflow(Vx, Vy, rho, mu, asound)

end


"""
    windturbineinflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

Compute relative wind velocity components along blade accounting for inflow conditions
and orientation of turbine.  See Documentation for angle definitions.

**Arguments**
- `Vhub::Float64`: freestream speed at hub (m/s)
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
function windturbineinflow(Vhub, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho, mu=1.0, asound=1.0)

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
    return Inflow(Vx, Vy, rho, mu, asound)

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
    loads(outputs)

Extract arrays for Np and Tp from the outputs.

This does not do any calculations.  It is simply a syntax shorthand as loads are usually what we mostly care about from the outputs.
"""
function loads(outputs)
    Np = getfield.(outputs, :Np)
    Tp = getfield.(outputs, :Tp)
    
    return Np, Tp
end


"""
    effectivewake(outputs)

Computes rotor wake velocities.

Note that the BEM methodology applies hub/tip losses to the loads rather than to the velocities.  
This is the most common way to implement a BEM, but it means that the raw velocities are misleading 
as they do not contain any hub/tip loss corrections.
To fix this we compute the effective hub/tip losses that would produce the same thrust/torque.
In other words:\n
CT = 4 a (1 - a) F = 4 a G (1 - a G)\n
This is solved for G, then multiplied against the wake velocities.
"""
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
    thrusttorque(rotor, sections, outputs)

integrate the thrust/torque across the blade, 
including 0 loads at hub/tip, using a trapezoidal rule.

**Arguments**
- `rotor::Rotor`: rotor object
- `sections::Array{Section, 1}`: section data along blade
- `outputs::Array{Outputs, 1}`: output data along blade

**Returns**
- `T::Float64`: thrust (along x-dir see Documentation)
- `Q::Float64`: torque (along x-dir see Documentation)
"""
function thrusttorque(rotor, sections, outputs)

    Np, Tp = loads(outputs)
    r = getfield.(sections, :r)

    # add hub/tip for complete integration.  loads go to zero at hub/tip.
    rfull = [rotor.Rhub; r; rotor.Rtip]
    Npfull = [0.0; Np; 0.0]
    Tpfull = [0.0; Tp; 0.0]

    # integrate Thrust and Torque (trapezoidal)
    thrust = Npfull*cos(rotor.precone)
    torque = Tpfull.*rfull*cos(rotor.precone)

    T = rotor.B * trapz(rfull, thrust)
    Q = rotor.B * trapz(rfull, torque)

    return T, Q
end

function thrusttorque(B, inputs::AbstractArray)

    num_radial = div(length(inputs)-3, 3)

    r = inputs[1:num_radial]
    Np = inputs[num_radial+1:2*num_radial]
    Tp = inputs[2*num_radial+1:3*num_radial]

    Rhub = inputs[3*num_radial+1]
    Rtip = inputs[3*num_radial+2]
    precone = inputs[3*num_radial+3]

    # Dummy values for stuff we don't need for the integrated loads. I guess I
    # could just reuse r, but whatever.
    array_dummy = similar(r)
    af_dummy = 0.

    rotor = Rotor(Rhub, Rtip, B, false, precone)
    sections = Section.(r, array_dummy, array_dummy, af_dummy)
    outputs = Outputs.(Np, Tp,
                       array_dummy,
                       array_dummy,
                       array_dummy,
                       array_dummy,
                       array_dummy,
                       array_dummy,
                       array_dummy,
                       array_dummy,
                       array_dummy)
    T, Q = thrusttorque(rotor, sections, outputs)

    return [T, Q]
    
end

function thrusttorque_partials(rotor, sections, Np, Tp)

    B = rotor.B
    function R(inputs)
        return thrusttorque(B, inputs)
    end

    num_radial = length(sections)
    x = zeros(typeof(sections[1].r), 3*num_radial+3)
    @. x[1:num_radial] = getfield(sections, :r)
    @. x[num_radial+1:2*num_radial] = Np
    @. x[2*num_radial+1:3*num_radial] = Tp
    x[3*num_radial+1] = rotor.Rhub
    x[3*num_radial+2] = rotor.Rtip
    x[3*num_radial+3] = rotor.precone

    return ForwardDiff.jacobian(R, x)
end


"""
ttodo
"""
function thrusttorque_azavg(rotor, sections, azinflows)

    T = 0.0
    Q = 0.0
    n = length(azinflows)

    for i = 1:n
        outputs = solve.(rotor, sections, azinflows[i])
        Tsub, Qsub = thrusttorque(rotor, sections, outputs)
        T += Tsub / n
        Q += Qsub / n
    end

    return T, Q
end


"""
(private) trapezoidal integration
"""
function trapz(x, y)  # integrate y w.r.t. x

    integral = 0.0
    for i = 1:length(x)-1
        integral += (x[i+1]-x[i])*0.5*(y[i] + y[i+1])
    end
    return integral
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
