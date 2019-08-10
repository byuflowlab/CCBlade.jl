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


export Rotor, OperatingPoint, Outputs
export af_from_file, af_from_data
export simple_op, windturbine_op
export solve, thrusttorque, nondim



# --------- structs -------------

"""
    Rotor(r, chord, theta, af, Rhub, Rtip, B, turbine, precone=0.0)

Define rotor geometry

**Arguments**
- `r::Array{Float64, 1}`: radial locations along turbine (Rhub < r < Rtip)
- `chord::Array{Float64, 1}`: corresponding local chord lengths
- `theta::Array{Float64, 1}`: corresponding twist angles (radians)
- `af::Array{function, 1}`: a function of the form: cl, cd = af(alpha, Re, Mach)
- `Rhub::Float64`: hub radius (along blade length)
- `Rtip::Float64`: tip radius (along blade length)
- `B::Int64`: number of blades
- `turbine::Bool`: true if turbine, false if propeller
- `precone::Float64`: precone angle
"""
struct Rotor{TF, TAF, TI, TB}
    
    r::AbstractArray{TF, 1}
    chord::AbstractArray{TF, 1}
    theta::AbstractArray{TF, 1}
    af::AbstractArray{TAF, 1}
    Rhub::TF
    Rtip::TF
    B::TI
    turbine::TB
    precone::TF

end

# convenience constructor for no precone
Rotor(r, chord, theta, af, Rhub, Rtip, B, turbine) = Rotor(r, chord, theta, af, Rhub, Rtip, B, turbine, 0.0)


"""
    OperatingPoint(Vx, Vy, pitch, rho, mu=1.0, asound=1.0)

Operation point for a rotor.  
The x direction is the axial direction, and y direction is the tangential direction in the rotor plane.  
See Documentation for more detail on coordinate systems.
Vx and Vy can vary both radially and in time (matrix of size [nr, nt]).  nr must match length(rotor.r)
whereas the fluid properties don't vary radially but can vary in time

**Arguments**
- `Vx::Array{Float64, 1}`: velocity in x-direction along blade
- `Vy::Array{Float64, 1}`: velocity in y-direction along blade
- `pitch::Float64`: pitch angle (rad).  defined same direction as twist.
- `rho::Float64`: fluid density
- `mu::Float64`: fluid dynamic viscosity (unused if Re not included in airfoil data)
- `asound::Float64`: fluid speed of sound (unused if Mach not included in airfoil data)
"""
struct OperatingPoint{TF}
    Vx::AbstractArray{TF, 1}
    Vy::AbstractArray{TF, 1}
    pitch::TF
    rho::TF
    mu::TF
    asound::TF
end

# convenience constructor when Re and Mach are not used.
OperatingPoint(Vx, Vy, pitch, rho) = OperatingPoint(Vx, Vy, pitch, rho, 1.0, 1.0) 


"""
    Outputs(Np, Tp, a, ap, u, v, phi, W, cl, cd, F)

Outputs from the BEM solver along the radius.

**Arguments**
- `Np::Array{Float64, 1}`: normal force per unit length
- `Tp::Array{Float64, 1}`: tangential force per unit length
- `a::Array{Float64, 1}`: axial induction factor
- `ap::Array{Float64, 1}`: tangential induction factor
- `u::Array{Float64, 1}`: axial induced velocity
- `v::Array{Float64, 1}`: tangential induced velocity
- `phi::Array{Float64, 1}`: inflow angle
- `alpha::Array{Float64, 1}`: angle of attack
- `W::Array{Float64, 1}`: inflow velocity
- `cl::Array{Float64, 1}`: lift coefficient
- `cd::Array{Float64, 1}`: drag coefficient
- `F::Array{Float64, 1}`: hub/tip loss correction
"""
struct Outputs{TF}

    Np::Array{TF, 1}
    Tp::Array{TF, 1}
    a::Array{TF, 1}
    ap::Array{TF, 1}
    u::Array{TF, 1}
    v::Array{TF, 1}
    phi::Array{TF, 1}
    alpha::Array{TF, 1}
    W::Array{TF, 1}
    cl::Array{TF, 1}
    cd::Array{TF, 1}
    cn::Array{TF, 1}
    ct::Array{TF, 1}
    F::Array{TF, 1}
    G::Array{TF, 1}

end

# convenience constructor
function Outputs(etype, nr)
    return Outputs(
        zeros(etype, nr), zeros(etype, nr), zeros(etype, nr), zeros(etype, nr),
        zeros(etype, nr), zeros(etype, nr), zeros(etype, nr), zeros(etype, nr),
        zeros(etype, nr), zeros(etype, nr), zeros(etype, nr), zeros(etype, nr),
        zeros(etype, nr), zeros(etype, nr), zeros(etype, nr)
    )
end


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

`af_from_file` calls this function indirectly.  Uses a cubic B-spline
(if the order of the data permits it).  A small amount of smoothing of
lift and drag coefficients is also applied to aid performance
for gradient-based optimization.

Returns a function of the form `cl, cd = func(alpha, Re, M)` although Re and M are currently ignored.
"""
function af_from_data(alpha, cl, cd, spl_k=3)

    # # TODO: update once smoothing is implemented: https://github.com/JuliaMath/Interpolations.jl/issues/254
    # afcl = CubicSplineInterpolation(alpha, cl)
    # afcd = CubicSplineInterpolation(alpha, cd)
    # af = AirfoilData(afcl, afcd)

    k = min(length(alpha)-1, spl_k)  # can't use cubic spline is number of entries in alpha is small

    # 1D interpolations for now.  ignoring Re dependence (which is very minor)
    afcl_1d = Dierckx.Spline1D(alpha, cl; k=k, s=0.1)
    afcd_1d = Dierckx.Spline1D(alpha, cd; k=k, s=0.001)

    afeval(alpha, Re, M) = afcl_1d(alpha), afcd_1d(alpha)  # ignore Re, M 

    return afeval
end



# ---------------------------------



# ------------ BEM core ------------------


"""
(private) residual function
modifies outputs if setoutputs=true
"""
function residual!(phi, rotor, inflow, outputs, idx, setoutputs=false)

    # unpack inputs
    r = rotor.r[idx]
    chord = rotor.chord[idx]
    theta = rotor.theta[idx]
    af = rotor.af[idx]
    Rhub = rotor.Rhub
    Rtip = rotor.Rtip
    B = rotor.B
    Vx = inflow.Vx[idx]
    Vy = inflow.Vy[idx]
    pitch = inflow.pitch
    rho = inflow.rho
    
    # check if turbine or propeller and change input sign if necessary
    setsign = rotor.turbine ? 1 : -1
    theta *= setsign
    pitch *= setsign
    Vx *= setsign

    # constants
    sigma_p = B*chord/(2.0*pi*r)
    sphi = sin(phi)
    cphi = cos(phi)

    # angle of attack
    alpha = phi - (theta + pitch)

    # Reynolds number
    W0 = sqrt(Vx^2 + Vy^2)  # ignoring induction, which is generally a very minor difference and only affects Reynolds/Mach number
    Re = rho * W0 * chord / inflow.mu

    # Mach number
    Mach = W0/inflow.asound  # also ignoring induction

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

    # # parameters used in Vx=0 and Vy=0 cases
    # k0 = cn*sigma_p/(4.0*F*sphi*cphi)
    # k0p = ct*sigma_p/(4.0*F*sphi*sphi)


    # --- solve for induced velocities ------
    # if isapprox(Vx, 0.0, atol=1e-6)

    #     u = sign(phi)*k0*Vy
    #     v = 0.0
    #     a = 0.0
    #     ap = 0.0

    # elseif isapprox(Vy, 0.0, atol=1e-6)
        
    #     u = 0.0
    #     v = k0p*abs(Vx)
    #     a = 0.0
    #     ap = 0.0
    
    # else

    if phi < 0
        k *= -1
    end

    if isapprox(k, -1.0, atol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
        return 1.0
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
        kp *= -1
    end

    if isapprox(kp, 1.0, atol=1e-6)  # state corresopnds to Vy=0, return any nonzero residual
        return 1.0
    end

    ap = kp/(1 - kp)
    v = ap * Vy


    # end

    # ------- residual function -------------
    # R = sin(phi)/(1 - a) - Vx/Vy*cos(phi)/(1 + ap)
    R = sin(phi)/(Vx - u) - cos(phi)/(Vy + v)


    # fill outputs
    if setoutputs

        # ------- loads ---------
        W2 = (Vx - u)^2 + (Vy + v)^2
        Np = cn*0.5*rho*W2*chord
        Tp = ct*0.5*rho*W2*chord

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

        outputs.Np[idx] = Np
        outputs.Tp[idx] = Tp
        outputs.a[idx] = a
        outputs.ap[idx] = ap
        outputs.u[idx] = u
        outputs.v[idx] = v
        outputs.phi[idx] = phi
        outputs.alpha[idx] = alpha
        outputs.W[idx] = sqrt(W2)
        outputs.cl[idx] = cl
        outputs.cd[idx] = cd
        outputs.cn[idx] = cn
        outputs.ct[idx] = ct
        outputs.F[idx] = F
        outputs.G[idx] = G
    end

    return R

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
    solve(rotor, section, inflow)

Solve the BEM equations for one section, with given inflow conditions, and rotor properties.
If multiple sections are to be solved (typical usage) then one can use broadcasting:
`solve.(sections, inflows, rotor)` where sections and inflows are arrays.

**Arguments**
- `rotor::Rotor`: rotor properties
- `op::OperatingPoint`: operating point

**Returns**
- `outputs::Outputs`: BEM output data including loads, induction factors, etc.
"""
function solve(rotor::Rotor, op::OperatingPoint)

    # parameters
    npts = 20  # number of discretization points to find bracket in residual solve

    # unpack
    setsign = rotor.turbine ? 1 : -1
    Vx = op.Vx * setsign  # TODO: ideally should do sign swap only in one place
    Vy = op.Vy
    # theta = rotor.theta * setsign
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

    #     startfrom90 = true  # start bracket search from 90 deg

    #     if Vx > 0 && abs(theta) < pi/2
    #         order = (q1, q3)
    #     elseif Vx < 0 && abs(theta) < pi/2
    #         order = (q2, q4)
    #     elseif Vx > 0 && abs(theta) > pi/2
    #         order = (q3, q1)
    #     else  # Vx < 0 && abs(theta) > pi/2
    #         order = (q4, q2)
    #     end

    # else  # normal case

    nr = length(rotor.r)
    
    # initialize outputs
    outputs = Outputs(eltype(rotor.r[1]), nr)

    for i = 1:nr

        startfrom90 = false

        if Vx[i] > 0 && Vy[i] > 0
            order = (q1, q2, q3, q4)
        elseif Vx[i] < 0 && Vy[i] > 0
            order = (q2, q1, q4, q3)
        elseif Vx[i] > 0 && Vy[i] < 0
            order = (q3, q4, q1, q2)
        else  # Vx[i] < 0 && Vy[i] < 0
            order = (q4, q3, q2, q1)
        end

        # end

        # ----- solve residual function ------

        

        # # wrapper to residual function to accomodate format required by fzero
        R(phi) = residual!(phi, rotor, op, outputs, i)

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

            # find bracket
            success, phiL, phiU = firstbracket(R, phimin, phimax, npts, backwardsearch)

            # once bracket is found, solve root finding problem and compute loads
            if success

                phistar = Roots.fzero(R, phiL, phiU)
                residual!(phistar, rotor, op, outputs, i, true)  # call once more to set outputs
                break
            end    
        end    
    end

    # it shouldn't get to this point.  if it does it means no solution was found
    # it will return empty outputs
    # alternatively, one could increase npts and try again
    
    return outputs
end


# function solve(rotor::Rotor, op::AbstractArray{OperatingPoint, 1})
#     n = length(op)
#     output_array = Array{Outputs, n}

#     for i = 1:n
#         output_array[i] = solve(rotor, op[i])
#     end
# end


# ------------ inflow ------------------



"""
    simple_op(Vinf, Omega, r, rho, mu=1.0, asound=1.0, precone=0.0)

Uniform inflow through rotor.  Returns an Inflow object.

**Arguments**
- `Vinf::Float`: freestream speed (m/s)
- `Omega::Float`: rotation speed (rad/s)
- `r::Float{Float64, 1}`: radial location where inflow is computed (m)
- `precone::Float64`: precone angle (rad)
- `rho::Float`: air density (kg/m^3)
- `pitch::Float`: pitch (rad)
- `mu::Float`: air viscosity (Pa * s)
- `asounnd::Float`: air speed of sound (m/s)
"""
function simple_op(Vinf, Omega, r, rho, pitch=0.0, mu=1.0, asound=1.0, precone=0.0)

    Vx = Vinf * cos(precone) * ones(length(r))
    Vy = Omega .* r * cos(precone)

    return OperatingPoint(Vx, Vy, pitch, rho, mu, asound)

end


"""
    windturbineinflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

Compute relative wind velocity components along blade accounting for inflow conditions
and orientation of turbine.  See Documentation for angle definitions.

**Arguments**
- `Vhub::Float64`: freestream speed at hub (m/s)
- `Omega::Float64`: rotation speed (rad/s)
- `pitch::Float64`: pitch angle (rad)
- `r::Array{Float64, 1}`: radial locations where inflow is computed (m)
- `precone::Float64`: precone angle (rad)
- `yaw::Float64`: yaw angle (rad)
- `tilt::Float64`: tilt angle (rad)
- `azimuth::Float64`: azimuth angle to evaluate at (rad)
- `hubHt::Float64`: hub height (m) - used for shear
- `shearExp::Float64`: power law shear exponent
- `rho::Float64`: air density (kg/m^3)
"""
function windturbine_op(Vhub, Omega, pitch, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho, mu=1.0, asound=1.0)

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
    heightFromHub = (y_az*sa .+ z_az*ca)*ct .- x_az*st

    # velocity with shear
    V = Vhub*(1 .+ heightFromHub/hubHt).^shearExp

    # transform wind to blade c.s.
    Vwind_x = V * ((cy*st*ca + sy*sa)*sc + cy*ct*cc)
    Vwind_y = V * (cy*st*sa - sy*ca)

    # wind from rotation to blade c.s.
    Vrot_x = -Omega*y_az*sc
    Vrot_y = Omega*z_az

    # total velocity
    Vx = Vwind_x .+ Vrot_x
    Vy = Vwind_y .+ Vrot_y

    # operating point
    return OperatingPoint(Vx, Vy, pitch, rho, mu, asound)

end

# """
# Convenience method for generating multiple windturbineinflow objects across multiple azimuth angles for later integration
# """
# function windturbineinflow_az(Vhub, Omega, pitch, r, precone, yaw, tilt, azimuth_array, hubHt, shearExp, rho, mu=1.0, asound=1.0)

#     naz = length(azimuth_array)
#     azinflows = Array{OperatingPoint}(undef, naz)
#     for i = 1:naz
#         azinflows[i] = windturbine_op(Vhub, Omega, pitch, r, precone, yaw, tilt, azimuth_array[i], hubHt, shearExp, rho)
#     end

#     return azinflows
# end


# -------------------------------------


# -------- convenience methods ------------

"""
    thrusttorque(rotor, sections, outputs)

integrate the thrust/torque across the blade, 
including 0 loads at hub/tip, using a trapezoidal rule.

**Arguments**
- `rotor::Rotor`: rotor object
- `outputs::Outputs`: output data along blade

**Returns**
- `T::Array{Float64, 1}`: thrust (along x-dir see Documentation). one for each time step
- `Q::Array{Float64, 1}`: torque (along x-dir see Documentation). one for each time step
"""
function thrusttorque(rotor::Rotor, outputs::Outputs)

    # add hub/tip for complete integration.  loads go to zero at hub/tip.
    rfull = [rotor.Rhub; rotor.r; rotor.Rtip]
    Npfull = [0.0; outputs.Np; 0.0]
    Tpfull = [0.0; outputs.Tp; 0.0]

    # integrate Thrust and Torque (trapezoidal)
    thrust = Npfull*cos(rotor.precone)
    torque = Tpfull.*rfull*cos(rotor.precone)

    T = rotor.B * trapz(rfull, thrust)
    Q = rotor.B * trapz(rfull, torque)

    return T, Q
end

function thrusttorque(rotor::Rotor, outputs::AbstractArray{Outputs{TF}, 1}) where TF <: Number

    T = 0.0
    Q = 0.0
    n = length(outputs)

    for i = 1:n
        Tsub, Qsub = thrusttorque(rotor, outputs[i])
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
