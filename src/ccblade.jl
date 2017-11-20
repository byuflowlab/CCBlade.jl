#=
Author: Andrew Ning

Blade Element Momentum Method for Propellers and Turbines
Allows for non-ideal conditions (reversed flow, no wind in one direction, etc.)

=#

module CCBlade

using Roots: fzero  # solve residual equation
using Dierckx  # cubic b-spline for airfoil cl/cd data
# using ForwardDiff

export AirfoilData, Rotor, Inflow
export readaerodyn, readafdata
export simpleInflow, windTurbineInflow, windTurbineInflowMultiple
export distributedLoads, thrusttorque, nondim

# TODO re-add AD gradients
# include("/Users/andrewning/Dropbox/BYU/repos/gradients/Smooth.jl")

# pretabulated cl/cd data
struct AirfoilData
    cl::Spline1D
    cd::Spline1D
end

# # data for a given BEM section
struct Rotor
    r#::Array{Float64, 1}
    chord#::Array{Float64, 1}
    theta#::Array{Float64, 1}
    af#::Array{AirfoilData, 1}
    Rhub#::Float64
    Rtip#::Float64
    B#::Int64
    precone#::Float64
end

# operating point for the turbine/propeller
struct Inflow
    Vx#::Array{Float64, 1}
    Vy#::Array{Float64, 1}
    rho#::Float64
end


function readaerodyn(filename)
    """
    read af files in aerodyn format.
    currently only reads one Reynolds number if multiple exist
    """

    alpha = Float64[]
    cl = Float64[]
    cd = Float64[]

    open(filename) do f

        # skip header
        for i = 1:13
            readline(f)
        end

        # read until EOT
        while true
            line = readline(f)
            if contains(line, "EOT")
                break
            end
            parts = split(line)
            push!(alpha, float(parts[1]))
            push!(cl, float(parts[2]))
            push!(cd, float(parts[3]))
        end
    end

    return readafdata(alpha, cl, cd)
end


function readafdata(alpha, cl, cd)
    """
    initialize airfoil directly from alpha, cl, cd data
    """

    k = min(length(alpha)-1, 3)  # can't use cubic spline is number of entries in alpha is small

    # 1D interpolations for now.  ignoring Re dependence (which is very minor)
    afcl = Dierckx.Spline1D(alpha*pi/180.0, cl; k=k, s=0.1)
    afcd = Dierckx.Spline1D(alpha*pi/180.0, cd; k=k, s=0.001)
    af = AirfoilData(afcl, afcd)

    return af
end


function airfoil(af::AirfoilData, alpha::Float64)

    cl = af.cl(alpha)
    cd = af.cd(alpha)

    return cl, cd
end


# function airfoil{T<:ForwardDiff.Dual}(af::AirfoilData, alpha::T)
#
#     a = GradEval.values(alpha)[1]
#     cl, cd = airfoil(af, a)
#
#     dclda = Dierckx.derivative(af.cl, a)
#     dcdda = Dierckx.derivative(af.cd, a)
#
#     cldual = GradEval.manualderiv(cl, alpha, dclda)
#     cddual = GradEval.manualderiv(cd, alpha, dcdda)
#
#     return cldual, cddual
# end


function simpleInflow(Vinf, Omega, r, precone, rho)

    Vx = Vinf * cos(precone) * ones(r)
    Vy = Omega * r * cos(precone)

    return Inflow(Vx, Vy, rho)

end

function windTurbineInflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

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
    y_az = zeros(r)  # could omit (the more general case allows for presweep so this is nonzero)

    # get section heights in wind-aligned coordinate system
    heightFromHub = (y_az*sa + z_az*ca)*ct - x_az*st

    # velocity with shear
    V = Vinf*(1 + heightFromHub/hubHt).^shearExp

    # transform wind to blade c.s.
    Vwind_x = V .* ((cy*st*ca + sy*sa)*sc + cy*ct*cc)
    Vwind_y = V .* (cy*st*sa - sy*ca)

    # wind from rotation to blade c.s.
    Vrot_x = -Omega*y_az.*sc
    Vrot_y = Omega*z_az

    # total velocity
    Vx = Vwind_x + Vrot_x
    Vy = Vwind_y + Vrot_y

    # operating point
    return Inflow(Vx, Vy, rho)

end

function windTurbineInflowMultiple(nsectors, Vinf, Omega, r, precone, yaw, tilt, hubHt, shearExp, rho)


    infs = Array{Inflow}(nsectors)

    for j = 1:nsectors
        azimuth = 2*pi*float(j)/nsectors
        infs[j] = windTurbineInflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)
    end

    return infs
end



# x = [r, chord, twist, Vx, Vy, Rhub, Rtip, rho]  (potential design variables or dependencies of design variables)
# p = [af, B]  (parameters)
function residualbase(phi, x, p)
    """
    residual function
    base calculations used in normal case, Vx=0 case, and Vy=0 case
    """

    # unpack variables for convenience
    r, chord, twist, Vx, Vy, Rhub, Rtip, rho = x
    af, B = p

    # constants
    sigma_p = B/2.0/pi*chord/r
    sphi = sin(phi)
    cphi = cos(phi)

    # angle of attack
    alpha = phi - twist

    # # Reynolds number (not currently used)
    # W_Re = sqrt(Vx^2 + Vy^2)  # ignoring induction, which is generally a very minor error and only affects Reynolds number
    # Re = rho * W_Re * chord / mu

    # airfoil cl/cd
    cl, cd = airfoil(af, alpha)

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
    k2 = cn*sigma_p/(4.0*F*cphi*cphi)
    kp2 = ct*sigma_p/(4.0*F*sphi*sphi)

    # force for unit W (because W varies between cases)
    Nunit = cn*0.5*rho*chord
    Tunit = ct*0.5*rho*chord

    return k, kp, F, k2, kp2, Nunit, Tunit

end


# x = [r, chord, twist, Vx, Vy, Rhub, Rtip, rho]
# p = [af, B]
function residual(phi, x, p)  #sec::Section)
    """residual for normal case"""

    k, kp, F, _, _, Nu, Tu = residualbase(phi, x, p)
    Vx = x[4]
    Vy = x[5]

    # -------- axial induction ----------
    if phi < 0
        k = -k
    end

    if isapprox(k, -1.0, atol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
        R = 1.0
        return R, 0.0, 0.0
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

    # -------- tangential induction ----------
    if Vx < 0
        kp = -kp
    end

    if isapprox(kp, 1.0, atol=1e-6)  # state corresopnds to Vy=0, return any nonzero residual
        R = 1.0
        return R, 0.0, 0.0
    end

    ap = kp/(1 - kp)


    # ------- residual function -------------
    R = sin(phi)/(1 - a) - Vx/Vy*cos(phi)/(1 + ap)

    # ------- loads ---------
    W2 = (Vx*(1-a))^2 + (Vy*(1+ap))^2
    Np = Nu*W2
    Tp = Tu*W2

    return R, Np, Tp
end



# x = [r, chord, twist, Vx, Vy, Rhub, Rtip, rho]
# p = [af, B]
function residualVx0(phi, x, p)
    """residual for Vx=0 case"""

    # base params
    _, _, _, k2, _, Nu, Tu = residualbase(phi, x, p)
    Vy = x[5]

    os = -sign(phi)

    if os*k2 < 0
        R = 1.0  # any nonzero residual

        Np = 0.0
        Tp = 0.0
    else
        # induced velocity
        u = os*sqrt(os*k2)*abs(Vy)

        # residual
        R = - sin(phi)/u - cos(phi)/Vy

        # loads
        W2 = u^2 + Vy^2
        Np = Nu*W2
        Tp = Tu*W2
    end


    return R, Np, Tp
end

# x = [r, chord, twist, Vx, Vy, Rhub, Rtip, rho]
# p = [af, B]
function residualVy0(phi, x, p)
    """residual for Vy=0 case"""

    # base params
    _, _, _, _, kp2, Nu, Tu = residualbase(phi, x, p)
    Vx = x[4]

    # induced velocity
    v = kp2*abs(Vx)

    # residual
    R = v*sin(phi) - Vx*cos(phi)

    # loads
    W2 = v^2 + Vx^2
    Np = Nu*W2
    Tp = Tu*W2

    return R, Np, Tp
end



function firstbracket(f, xmin, xmax, n, backwardsearch=false)
    """Find a bracket for the root closest to xmin by subdividing
    interval (xmin, xmax) into n intervals

    Returns found, xl, xu.
    If found = true a bracket was found between (xl, xu)
    """

    xvec = linspace(xmin, xmax, n)
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


function distributedLoads(rotor::Rotor, inflow::Inflow, turbine::Bool)
    """
    Compute the distributed loads along blade at specified condition
    """

    # check if propeller
    swapsign = turbine ? 1 : -1

    # parameters
    npts = 20  # number of discretization points to find bracket in residual solve

    # quadrants
    epsilon = 1e-6
    q1 = [epsilon, pi/2]
    q2 = [-pi/2, -epsilon]
    q3 = [pi/2, pi-epsilon]
    q4 = [-pi+epsilon, -pi/2]

    # initialize arrays
    n = length(rotor.r)
    Np = zeros(n)
    Tp = zeros(n)

    # if isa(rotor.r[1], ForwardDiff.Dual)  # hack for now...I shouldn't have to do this.
    #     nd = length(ForwardDiff.partials(rotor.r[1]))
    #     Np = Array{ForwardDiff.Dual{nd,Float64}}(n)  # an array of dual numbers
    #     Tp = Array{ForwardDiff.Dual{nd,Float64}}(n)  # an array of dual numbers
    #
    # else
    #     Np = zeros(n)
    #     Tp = zeros(n)
    # end

    for i = 1:n  # iterate across blade

        # setup
        twist = swapsign*rotor.theta[i] #  + op.pitch  TODO: pitch is just part of theta specificiation
        Vx = swapsign*inflow.Vx[i]
        Vy = inflow.Vy[i]
        startfrom90 = false  # start bracket search from 90 deg instead of 0 deg.

        # Vx = 0 and Vy = 0
        if isapprox(Vx, 0.0, atol=1e-6) && isapprox(Vy, 0.0, atol=1e-6)
            Np[i] = 0.0; Tp[i] = 0.0
            break

        # Vx = 0
    elseif isapprox(Vx, 0.0, atol=1e-6)

            resid = residualVx0

            if Vy > 0 && twist > 0
                order = (q1, q2)
            elseif Vy > 0 && twist < 0
                order = (q2, q1)
            elseif Vy < 0 && twist > 0
                order = (q3, q4)
            else  # Vy < 0 && twist < 0
                order = (q4, q3)
            end

        # Vy = 0
    elseif isapprox(Vy, 0.0, atol=1e-6)

            resid = residualVy0
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

            resid = residual

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

        # wrapper to residual function to accomodate format required by fzero
        x = [rotor.r[i], rotor.chord[i], twist, Vx, Vy, rotor.Rhub, rotor.Rtip, inflow.rho]
        p = [rotor.af[i], rotor.B]
        function func(x, phi)
            zero, Npinner, Tpinner = resid(phi, x, p)
            return [Npinner; Tpinner], zero
        end

        function R(phi)
            zero, _, _ = resid(phi, x, p)
            return zero
        end

        success = false
        for j = 1:length(order)  # quadrant orders.  In most cases it should find root in first quadrant searched.
            phimin = order[j][1]
            phimax = order[j][2]

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

                # f = Smooth.fzerod(func, fzero, x, phiL, phiU)
                #
                # Np[i] = f[1]
                # Tp[i] = f[2]

                phistar = fzero(R, phiL, phiU)
                _, Np[i], Tp[i] = resid(phistar, x, p)

                break
            end

            # if no solution found, it just returns zeros for this section, although this really shouldn't get here
            # alternatively, one could increase npts and try again
        end
    end

    Tp = swapsign * Tp  # reverse sign for propellers

    return Np, Tp
end




function thrusttorqueintegrate(Rhub, r, Rtip, precone, Np, Tp)
    """integrate the thrust/torque across the blade, including 0 loads at hub/tip"""

    # add hub/tip for complete integration.  loads go to zero at hub/tip.
    rfull = [Rhub; r; Rtip]
    Npfull = [0.0; Np; 0.0]
    Tpfull = [0.0; Tp; 0.0]

    # integrate Thrust and Torque (trapezoidal)
    thrust = Npfull*cos(precone)
    torque = Tpfull.*rfull*cos(precone)

    T = trapz(rfull, thrust)
    Q = trapz(rfull, torque)

    return T, Q
end


function trapz(x::Array{Float64,1}, y::Array{Float64,1})  # integrate y w.r.t. x
    """trapezoidal integration"""

    integral = 0.0
    for i = 1:length(x)-1
        integral += (x[i+1]-x[i])*0.5*(y[i] + y[i+1])
    end
    return integral
end


function thrusttorque(rotor::Rotor, inflow::Array{Inflow, 1}, turbine::Bool)  #, nsectors::Int64)
    """one operating point for each sector"""

    nsectors = length(inflow)  # number of sectors (should be evenly spaced)
    T = 0.0
    Q = 0.0

    # coarse integration - rectangle rule
    for j = 1:nsectors  # integrate across azimuth

        Np, Tp = distributedLoads(rotor, inflow[j], turbine)

        Tsub, Qsub = thrusttorqueintegrate(rotor.Rhub, rotor.r, rotor.Rtip, rotor.precone, Np, Tp)

        T += rotor.B * Tsub / nsectors
        Q += rotor.B * Qsub / nsectors
    end

    return T, Q

end


function nondim(T, Q, Vhub, Omega, rho, Rtip, precone, turbine)
    """nondimensionalize"""

    P = Q * Omega
    Rp = Rtip*cos(precone)

    if turbine

        q = 0.5 * rho * Vhub^2
        A = pi * Rp^2

        CP = P / (q * A * Vhub)
        CT = T / (q * A)
        CQ = Q / (q * Rp * A)

        return CP, CT, CQ

    else

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

    end

end


end  # module
