using Roots

immutable Section  # TODO: change to struct after julia 0.6
    r::Float64
    chord::Float64
    twist::Float64
    Rhub::Float64
    Rtip::Float64
    af::AirfoilData
    B::Int64
    Vx::Float64
    Vy::Float64
end

immutable Rotor
    r::Array{Float64, 1}
    chord::Array{Float64, 1}
    theta::Array{Float64, 1}
    af::Array{AirfoilData, 1}
    Rhub::Float64
    Rtip::Float64
    B::Int64
    # precone::Float64
    # tilt::Float64
    # yaw::Float64
    # hubHt::Float64
end

# operating point for the turbine/propeller
immutable OperatingPoint
    Vinf::Float64
    Omega::Float64
    pitch::Float64
    azimuth::Float64
end


function firstbracket(f, xmin, xmax, n)

    xvec = linspace(xmin, xmax, n)
    fprev = f(xvec(1))
    for i = 2:n
        fnext = fvec(i)
        if fprev*fnext < 0  # bracket found
            return true, xvec(i-1), xvec(i)
        end
        fprev = fnext
    end

    return false, 0.0, 0.0

end


function distributedLoads(rotor::Rotor, op::OperatingPoint)

    # parameters
    npts = 20

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

    for i = 1:n

        # setup
        twist = rotor.theta[i] + op.pitch
        Vx = Vxvec[i]
        Vy = Vyvec[i]
        sec = Section(rotor.r[i], rotor.chord[i], twist, rotor.Rhub,
            rotor.Rtip, rotor.af[i], rotor.B, Vx, Vy)

        if isapprox(Vx, 0.0, rtol=1e-6) && isapprox(Vy, 0.0, rtol=1e-6)
            Np[i] = 0.0; Tp[i] = 0.0
            break

        elseif isapprox(Vx, 0.0, rtol=1e-6)

            R(phi) = residualVx0(phi, sec)

            if Vy > 0 && twist > 0
                order = (q1, q2)
            elseif Vy > 0 && twist < 0
                order = (q2, q1)
            elseif Vy < 0 && twist > 0
                order = (q3, q4)
            else  # Vy < 0 && twist < 0
                order = (q4, q3)
            end

        elseif isapprox(Vy, 0.0, rtol=1e-6)

            R(phi) = residualVy0(phi, sec)

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

            R(phi) = residualbase(phi, sec)

            if Vx > 0 && Vy > 0
                order = (q1, q2, q3, q4)
            elseif Vx < 0 && Vy > 0
                order = (q2, q1, q4, q3)
            elseif Vx > 0 && Vy < 0
                order = (q3, q4, q1, q2)
            else  # Vx < 0 && Vy < 0
                order = (q4, q3, q2, q1)

        for j = 1:length(order)
            phimin = order[j][1]
            phimax = order[j][2]
            success, phiL, phiU = firstbracket(R, phimin, phimax, npts)
            if success
                break
            end
        end
        if !success
            # no solution found.  could increase npts and try again or just return zeros
            Np[i] = 0.0; Tp[i] = 0.0
            break
        end

        phistar = fzero(R, phiL, phiU)

        _, _, _, _, _, Np[i], Tp[i] = residualbase(phistar, sec)


end


function residualbase(phi::Float64, sec::Section)

    r = sec.r
    chord = sec.chord
    twist = sec.twist
    Rhub = sec.Rhub
    Rtip = sec.Rtip
    B = sec.B
    Vx = sec.Vx
    Vy = sec.Vy

    # constants
    sigma_p = B/2.0/pi*chord/r
    sphi = sin(phi)
    cphi = cos(phi)

    # angle of attack
    alpha = phi - twist

    # Reynolds number (not currently used)
    W_Re = sqrt(Vx^2 + Vy^2)  # ignoring induction, which is generally a very minor error and only affects Reynolds number
    Re = rho * W_Re * chord / mu

    # airfoil cl/cd
    cl = sec.af.cl[alpha]
    cd = sec.af.cd[alpha]

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

    k2 = cn*sigma_p/(4.0*F*cphi*cphi)
    kp2 = ct*sigma_p/(4.0*F*sphi*sphi)

    # force for unit unit W
    Nunit = B*cn*0.5*rho*chord
    Tunit = B*ct*0.5*rho*chord

    return k, kp, F, k2, kp2, Nunit, Tunit

end



function residual(phi::Float64, sec::Section)

    k, kp, F, _, _ Nu, Tu = residualbase(phi, sec)
    Vx = sec.Vx
    Vy = sec.Vy

    # -------- axial induction ----------
    if phi < 0
        k = -k
    end

    if isapprox(k, -1.0, rtol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
        R = 1.0
        return R
    end

    if k <= 2.0/3  # momentum region
        a = k/(1 + k)

    else  # empirical region
        g1 = 2.0*F*k - (10.0/9-F)
        g2 = 2.0*F*k - (4.0/3-F)*F
        g3 = 2.0*F*k - (25.0/9-2*F)

        if isapprox(g3, 0.0, rtol=1e-6)  # avoid singularity
            a = 1.0 - 1.0/(2.0(sqrt(g2))
        else
            a = (g1 - sqrt(g2)) / g3
        end
    end

    # -------- tangential induction ----------
    if Vx < 0
        kp = -kp
    end

    if isapprox(kp, 1.0, rtol=1e-6)  # state corresopnds to Vy=0, return any nonzero residual
        R = 1.0
        return R
    end

    ap = kp/(1 - kp)


    # ------- residual function -------------
    R = sin(phi)/(1 - a) - Vx/Vy*cos(phi)/(1 + ap)

    # ------- loads ---------
    W = sqrt((Vx*(1-a))^2 + (Vy*(1+ap))^2)
    Np = Nu*W^2
    Tp = Tu*W^2

    return R, Np, Tp
end



function residualVx0(phi::Float64, sec::Section)

    # base params
    _, _, _, k2, _, Nu, Tu = residualbase(phi, sec)
    Vy = sec.Vy

    # induced velocity
    s = sign(phi)
    u = s*sqrt(s*k2)*abs(Vy)

    # residual
    R = - sin(phi)/u - cos(phi)/Vy

    # loads
    W = sqrt(u^2 + Vy^2)
    Np = Nu*W^2
    Tp = Tu*W^2

    return R, Np, Tp
end

function residualVy0(phi::Float64, sec::Section)

    # base params
    _, _, _, _, kp2, Nu, Tu = residualbase(phi, r, chord, twist, Rhub, Rtip, af, B, Vx, Vy)
    Vx = sec.Vx

    # induced velocity
    v = kp2*abs(Vx)

    # residual
    R = sin(phi)/Vx - cos(phi)/v

    # loads
    W = sqrt(v^2 + Vx^2)
    Np = Nu*W^2
    Tp = Tu*W^2

    return R, Np, Tp
end
