#=
Author: Andrew Ning

A general blade element momentum (BEM) method for propellers/fans and turbines.

Some unique features:
- a simple yet very robust solution method ideal for use with optimization
- designed for compatibility with algorithmic differentiation tools
- allows for arbitrary inflow conditions, including reversed flow, hover, etc.
- convenience methods for common wind turbine inflow scenarios

=#

module CCBlade

import FLOWMath
using ImplicitAD
using StructArrays: StructArray

export Rotor, Section, OperatingPoint, Outputs
export simple_op, windturbine_op
export solve, thrusttorque, nondim


include("airfoils.jl")  # all the code related to airfoil data

# --------- structs -------------

"""
    Rotor(Rhub, Rtip, B; precone=0.0, turbine=false,
        mach=nothing, re=nothing, rotation=nothing, tip=PrandtlTipHub())

Parameters defining the rotor (apply to all sections).

**Arguments**
- `Rhub::Float64`: hub radius (along blade length)
- `Rtip::Float64`: tip radius (along blade length)
- `B::Int64`: number of blades
- `precone::Float64`: precone angle
- `turbine::Bool`: true if using wind turbine conventions
- `mach::MachCorrection`: correction method for Mach number
- `re::ReCorrection`: correction method for Reynolds number
- `rotation::RotationCorrection`: correction method for blade rotation
- `tip::TipCorrection`: correction method for hub/tip loss
"""
struct Rotor{TF, TI, TB,
        T1 <: Union{Nothing, MachCorrection}, T2 <: Union{Nothing, ReCorrection},
        T3 <: Union{Nothing, RotationCorrection}, T4 <: Union{Nothing, TipCorrection}}
    Rhub::TF
    Rtip::TF
    B::TI
    precone::TF
    turbine::TB
    mach::T1
    re::T2
    rotation::T3
    tip::T4
end

# convenience constructor with keyword parameters
Rotor(Rhub, Rtip, B; precone=0.0, turbine=false, mach=nothing, re=nothing, rotation=nothing, tip=PrandtlTipHub()
    ) = Rotor(Rhub, Rtip, B, precone, turbine, mach, re, rotation, tip)


"""
    Section(r, chord, theta, af)

Define sectional properties for one station along rotor

**Arguments**
- `r::Float64`: radial location along blade
- `chord::Float64`: corresponding local chord length
- `theta::Float64`: corresponding twist angle (radians)
- `af::Function or AFType`: if function form is: `cl, cd = af(alpha, Re, Mach)`
"""
struct Section{TF, TAF}
    r::TF
    chord::TF
    theta::TF
    af::TAF
end

# promote to same type, e.g., duals
function Section(r, chord, theta, af)
    r, chord, theta = promote(r, chord, theta)
    return Section(r, chord, theta, af)
end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}, ::Type{Section{TF,TAF}}) where {N,TF,TAF}
    return StructArray{Section{TF,TAF}}(undef, size(bc))
end

"""
    OperatingPoint(Vx, Vy, rho; pitch=0.0, mu=1.0, asound=1.0)

Operation point for a rotor.
The x direction is the axial direction, and y direction is the tangential direction in the rotor plane.
See Documentation for more detail on coordinate systems.
`Vx` and `Vy` vary radially at same locations as `r` in the rotor definition.

**Arguments**
- `Vx::Float64`: velocity in x-direction along blade
- `Vy::Float64`: velocity in y-direction along blade
- `pitch::Float64`: pitch angle (radians)
- `rho::Float64`: fluid density
- `mu::Float64`: fluid dynamic viscosity (unused if Re not included in airfoil data)
- `asound::Float64`: fluid speed of sound (unused if Mach not included in airfoil data)
"""
struct OperatingPoint{TF}
    Vx::TF
    Vy::TF
    rho::TF
    pitch::TF
    mu::TF
    asound::TF
end

# promote to same type, e.g., duals
OperatingPoint(Vx, Vy, rho, pitch, mu, asound) = OperatingPoint(promote(Vx, Vy, rho, pitch, mu, asound)...)


# convenience constructor when Re and Mach are not used.
OperatingPoint(Vx, Vy, rho; pitch=zero(rho), mu=one(rho), asound=one(rho)) = OperatingPoint(Vx, Vy, rho, pitch, mu, asound)

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}, ::Type{OperatingPoint{TF}}) where {N,TF}
    return StructArray{OperatingPoint{TF}}(undef, size(bc))
end

"""
    Outputs(Np, Tp, a, ap, u, v, phi, alpha, W, cl, cd, cn, ct, F, G)

Outputs from the BEM solver along the radius.

**Arguments**
- `Np::Float64`: normal force per unit length
- `Tp::Float64`: tangential force per unit length
- `a::Float64`: axial induction factor
- `ap::Float64`: tangential induction factor
- `u::Float64`: axial induced velocity
- `v::Float64`: tangential induced velocity
- `phi::Float64`: inflow angle
- `alpha::Float64`: angle of attack
- `W::Float64`: inflow velocity
- `cl::Float64`: lift coefficient
- `cd::Float64`: drag coefficient
- `cn::Float64`: normal force coefficient
- `ct::Float64`: tangential force coefficient
- `F::Float64`: hub/tip loss correction
- `G::Float64`: effective hub/tip loss correction for induced velocities: `u = Vx * a * G, v = Vy * ap * G`
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

# promote to same type, e.g., duals
Outputs(Np, Tp, a, ap, u, v, phi, alpha, W, cl, cd, cn, ct, F, G) = Outputs(promote(Np, Tp, a, ap, u, v, phi, alpha, W, cl, cd, cn, ct, F, G)...)

# convenience constructor to initialize
Outputs() = Outputs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}, ::Type{Outputs{TF}}) where {N,TF}
    return StructArray{Outputs{TF}}(undef, size(bc))
end

# -------------------------------




# ------------ BEM core ------------------

"""
(private) residual function
"""
function residual_and_outputs(phi, x, p)  #rotor, section, op)

    # unpack inputs
    r, chord, theta, Rhub, Rtip, Vx, Vy, rho, pitch, mu, asound = x  # variables
    af, B, turbine, re_corr, mach_corr, rotation_corr, tip_corr = p  # parameters

    # constants
    sigma_p = B*chord/(2.0*pi*r)
    sphi = sin(phi)
    cphi = cos(phi)

    # angle of attack
    alpha = (theta + pitch) - phi

    # Reynolds/Mach number
    W0 = sqrt(Vx^2 + Vy^2)  # ignoring induction, which is generally a very minor difference and only affects Reynolds/Mach number
    Re = rho * W0 * chord / mu
    Mach = W0/asound  # also ignoring induction

    # airfoil cl/cd
    if turbine
        cl, cd = afeval(af, -alpha, Re, Mach)
        cl *= -1
    else
        cl, cd = afeval(af, alpha, Re, Mach)
    end

    # airfoil corrections
    if !isnothing(re_corr)
        cl, cd = re_correction(re_corr, cl, cd, Re)
    end
    if !isnothing(mach_corr)
        cl, cd = mach_correction(mach_corr, cl, cd, Mach)
    end
    if !isnothing(rotation_corr)
        cl, cd = rotation_correction(rotation_corr, cl, cd, chord/r, r/Rtip, Vy/Vx*Rtip/r, alpha, phi)
    end

    # resolve into normal and tangential forces
    cn = cl*cphi - cd*sphi
    ct = cl*sphi + cd*cphi

    # hub/tip loss
    F = 1.0
    if !isnothing(tip_corr)
        F = tip_correction(tip_corr, r, Rhub, Rtip, phi, B)
    end

    # sec parameters
    k = cn*sigma_p/(4.0*F*sphi*sphi)
    kp = ct*sigma_p/(4.0*F*sphi*cphi)

    # --- solve for induced velocities ------
    if isapprox(Vx, 0.0, atol=1e-6)

        u = sign(phi)*kp*cn/ct*Vy
        v = zero(phi)
        a = zero(phi)
        ap = zero(phi)
        R = sign(phi) - k

    elseif isapprox(Vy, 0.0, atol=1e-6)

        u = zero(phi)
        v = k*ct/cn*abs(Vx)
        a = zero(phi)
        ap = zero(phi)
        R = sign(Vx) + kp

    else

        if phi < 0
            k *= -1
        end

        if isapprox(k, 1.0, atol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
            return 1.0, Outputs()
        end

        if k >= -2.0/3  # momentum region
            a = k/(1 - k)

        else  # empirical region. Not Buhl's correction but instead uses Buhl with F = 1 then multiplied by F.
            # (Buhl(F = 1)*F).  The original method does not force CT -> 0 as f->0.  This can be problematic if
            # using CT/CP directly as a design variables.  Suggestion courtesy of Kenneth Lønbæk.
            g1 = 2*k + 1.0/9
            g2 = -2*k - 1.0/3
            g3 = -2*k - 7.0/9
            a = (g1 + sqrt(g2)) / g3
        end

        u = a * Vx

        # -------- tangential induction ----------
        if Vx < 0
            kp *= -1
        end

        if isapprox(kp, -1.0, atol=1e-6)  # state corresopnds to Vy=0, return any nonzero residual
            return 1.0, Outputs()
        end

        ap = kp/(1 + kp)
        v = ap * Vy


        # ------- residual function -------------
        R = sin(phi)/(1 + a) - Vx/Vy*cos(phi)/(1 - ap)
    end

    # ------- loads ---------
    W = sqrt((Vx + u)^2 + (Vy - v)^2)
    Np = cn*0.5*rho*W^2*chord
    Tp = ct*0.5*rho*W^2*chord

    # The BEM methodology applies hub/tip losses to the loads rather than to the velocities.
    # This is the most common way to implement a BEM, but it means that the raw velocities are misleading
    # as they do not contain any hub/tip loss corrections.
    # To fix this we compute the effective hub/tip losses that would produce the same thrust/torque.
    # In other words:
    # CT = 4 a (1 + a) F = 4 a G (1 + a G)\n
    # This is solved for G, then multiplied against the wake velocities.

    if isapprox(Vx, 0.0, atol=1e-6)
        G = sqrt(F)
    elseif isapprox(Vy, 0.0, atol=1e-6)
        G = F
    else
        G = (-1.0 + sqrt(1.0 + 4*a*(1.0 + a)*F))/(2*a)
    end
    u *= G
    v *= G

    if turbine
        return R, Outputs(-Np, -Tp, -a, -ap, -u, -v, phi, -alpha, W, -cl, cd, -cn, -ct, F, G)
    else
        return R, Outputs(Np, Tp, a, ap, u, v, phi, alpha, W, cl, cd, cn, ct, F, G)
    end

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
    solve(rotor, section, op; npts=10, forcebackwardsearch=false, epsilon_everywhere=false)

Solve the BEM equations for given rotor geometry and operating point.

**Arguments**
- `rotor::Rotor`: rotor properties
- `section::Section`: section properties
- `op::OperatingPoint`: operating point
- `npts::Int = 10`: number of discretization points for `phi` state variable, used to find bracket for residual solve
- `forcebackwardsearch::Bool = false`: if true, force bracket search from high `phi` values to low, otherwise let `solve` decide
- `epsilon_everywhere::Bool = false`: if true, don't evaluate at intersections of `phi` quadrants (`pi/2`, `-pi/2`, etc.)
- `implicitad_option=true`: if true, uses ImplicitAD to compute derivatives around solver when using AD; if false, bypasses ImplicitAD

**Returns**
- `outputs::Outputs`: BEM output data including loads, induction factors, etc.
"""
function solve(rotor, section, op; npts=10, forcebackwardsearch=false, epsilon_everywhere=false, implicitad_option=true)

    # error handling
    if typeof(section) <: AbstractVector
        error("You passed in an vector for section, but this function does not accept an vector.\nProbably you intended to use broadcasting (notice the dot): solve.(Ref(rotor), sections, ops)")
    end

    # check if we are at hub/tip
    if isapprox(section.r, rotor.Rhub, atol=1e-6) || isapprox(section.r, rotor.Rtip, atol=1e-6)
        return Outputs()  # no loads at hub/tip
    end

    # parameters
    # npts = 10  # number of discretization points to find bracket in residual solve

    # unpack
    Vx = op.Vx
    Vy = op.Vy
    theta = section.theta + op.pitch

    # ---- determine quadrants based on case -----
    Vx_is_zero = isapprox(Vx, 0.0, atol=1e-6)
    Vy_is_zero = isapprox(Vy, 0.0, atol=1e-6)

    # quadrants
    epsilon = 1e-6
    if epsilon_everywhere
        q1 = [epsilon, pi/2-epsilon]
        q2 = [-pi/2+epsilon, -epsilon]
        q3 = [pi/2+epsilon, pi-epsilon]
        q4 = [-pi+epsilon, -pi/2-epsilon]
    else
        q1 = [epsilon, pi/2]
        q2 = [-pi/2, -epsilon]
        q3 = [pi/2, pi-epsilon]
        q4 = [-pi+epsilon, -pi/2]
    end

    if Vx_is_zero && Vy_is_zero
        return Outputs()

    elseif Vx_is_zero

        startfrom90 = false  # start bracket at 0 deg.

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
        else  # Vx[i] < 0 && Vy[i] < 0
            order = (q4, q3, q2, q1)
        end

    end

    # ----- solve residual function ------
    # pull out first argument
    residual(phi, x, p) = residual_and_outputs(phi, x, p)[1]

    # package up variables and parameters for residual
    xv = [section.r, section.chord, section.theta, rotor.Rhub, rotor.Rtip, op.Vx, op.Vy, op.rho, op.pitch, op.mu, op.asound]
    pv = (section.af, rotor.B, rotor.turbine, rotor.re, rotor.mach, rotor.rotation, rotor.tip)

    success = false
    for j = 1:length(order)  # quadrant orders.  In most cases it should find root in first quadrant searched.
        phimin, phimax = order[j]

        if forcebackwardsearch
            backwardsearch = true
        else
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
        end

        # find bracket
        success, phiL, phiU = firstbracket(phi -> residual(phi, xv, pv), phimin, phimax, npts, backwardsearch)

        function solve(x, p)
            phistar, _ = FLOWMath.brent(phi -> residual(phi, x, p), phiL, phiU)
            return phistar
        end

        # once bracket is found, solve root finding problem and compute loads
        if success
            if implicitad_option
                phistar = implicit(solve, residual, xv, pv)
            else
                phistar = solve(xv, pv)
            end
            _, outputs = residual_and_outputs(phistar, xv, pv)
            return outputs
        end
    end

    # it shouldn't get to this point.  if it does it means no solution was found
    # it will return empty outputs
    # alternatively, one could increase npts and try again

    @warn "Invalid data (likely) for this section.  Zero loading assumed."
    return Outputs()
end


# ------------ inflow ------------------



"""
    simple_op(Vinf, Omega, r, rho; pitch=0.0, mu=1.0, asound=1.0, precone=0.0)

Uniform inflow through rotor.  Returns an OperatingPoint object.

**Arguments**
- `Vinf::Float`: freestream speed (m/s)
- `Omega::Float`: rotation speed (rad/s)
- `r::Float`: radial location where inflow is computed (m)
- `rho::Float`: air density (kg/m^3)
- `pitch::Float`: pitch angle (rad)
- `mu::Float`: air viscosity (Pa * s)
- `asounnd::Float`: air speed of sound (m/s)
- `precone::Float`: precone angle (rad)
"""
function simple_op(Vinf, Omega, r, rho; pitch=zero(rho), mu=one(rho), asound=one(rho), precone=zero(Vinf))

    # error handling
    if typeof(r) <: AbstractVector
        error("You passed in an vector for r, but this function does not accept an vector.\nProbably you intended to use broadcasting")
    end

    Vx = Vinf * cos(precone)
    Vy = Omega * r * cos(precone)

    return OperatingPoint(Vx, Vy, rho, pitch, mu, asound)

end


"""
    windturbine_op(Vhub, Omega, pitch, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho, mu=1.0, asound=1.0)

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
- `azimuth::Float64`: azimuth angle to evaluate at (rad)
- `hubHt::Float64`: hub height (m) - used for shear
- `shearExp::Float64`: power law shear exponent
- `rho::Float64`: air density (kg/m^3)
- `mu::Float64`: air viscosity (Pa * s)
- `asound::Float64`: air speed of sound (m/s)
"""
function windturbine_op(Vhub, Omega, pitch, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho, mu=one(rho), asound=one(rho))
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
    return OperatingPoint(Vx, Vy, rho, pitch, mu, asound)

end

# -------------------------------------


# -------- convenience methods ------------

"""
    thrusttorque(rotor, sections, outputs::AbstractVector{TO}) where TO

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
function thrusttorque(rotor, sections, outputs::AbstractVector{TO}) where TO

    # add hub/tip for complete integration.  loads go to zero at hub/tip.
    rvec = [s.r for s in sections]
    rfull = [rotor.Rhub; rvec; rotor.Rtip]
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
    thrusttorque(rotor, sections, outputs::AbstractMatrix{TO}) where TO

Integrate the thrust/torque across the blade given an array of output data.
Generally used for azimuthal averaging of thrust/torque.
`outputs[i, j]` corresponds to `sections[i], azimuth[j]`.  Integrates across azimuth
"""
function thrusttorque(rotor, sections, outputs::AbstractMatrix{TO}) where TO

    T = 0.0
    Q = 0.0
    nr, naz = size(outputs)

    for j = 1:naz
        Tsub, Qsub = thrusttorque(rotor, sections, outputs[:, j])
        T += Tsub / naz
        Q += Qsub / naz
    end

    return T, Q
end



"""
    nondim(T, Q, Vhub, Omega, rho, rotor, rotortype)

Nondimensionalize the outputs.

**Arguments**
- `T::Float64`: thrust (N)
- `Q::Float64`: torque (N-m)
- `Vhub::Float64`: hub speed used in turbine normalization (m/s)
- `Omega::Float64`: rotation speed used in propeller normalization (rad/s)
- `rho::Float64`: air density (kg/m^3)
- `rotor::Rotor`: rotor object
- `rotortype::String`: normalization type

**Returns**

`if rotortype == "windturbine"`
- `CP::Float64`: power coefficient
- `CT::Float64`: thrust coefficient
- `CQ::Float64`: torque coefficient

`if rotortype == "propeller"`
- `eff::Float64`: efficiency
- `CT::Float64`: thrust coefficient
- `CQ::Float64`: torque coefficient

`if rotortype == "helicopter"`
- `FM::Float64`: figure of merit
- `CT::Float64`: thrust coefficient
- `CQ or CP::Float64`: torque/power coefficient (they are identical)
"""
function nondim(T, Q, Vhub, Omega, rho, rotor, rotortype)

    P = Q * Omega
    Rp = rotor.Rtip*cos(rotor.precone)

    if rotortype == "windturbine"  # wind turbine normalizations

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
            eff = 0.0  # creating drag not thrust
        else
            eff = T*Vhub/P
        end
        CT = T / (rho * n^2 * Dp^4)
        CQ = Q / (rho * n^2 * Dp^5)

        return eff, CT, CQ

    elseif rotortype == "helicopter"

        A = pi * Rp^2

        CT = T / (rho * A * (Omega*Rp)^2)
        CP = P / (rho * A * (Omega*Rp)^3)  # note that CQ = CP
        if CT < 0
            FM = 0.0 # creating drag not thrust
        else
            FM = CT^(3.0/2)/(sqrt(2)*CP)
        end

        return FM, CT, CP
    end

end


end  # module
