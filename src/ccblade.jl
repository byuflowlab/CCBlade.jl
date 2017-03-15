



function residualbase(phi::Float64, r::Float64, chord::Float64, twist::Float64,
    Rhub::Float64, Rtip::Float64, af::AirfoilData, B::Int64,
    Vx::Float64, Vy::Float64)

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
    cl = af.cl[alpha]
    cd = af.cd[alpha]

    # resolve into normal and tangential forces
    cn = cl*cphi + cd*sphi
    ct = cl*sphi - cd*cphi

    # Prandtl's tip and hub loss factor
    factortip = B/2.0*(Rtip - r)/(r*abs(sphi))
    Ftip = 2.0/pi*acos(exp(-factortip))
    factorhub = B/2.0*(r - Rhub)/(Rhub*abs(sphi))
    Fhub = 2.0/pi*acos(exp(-factorhub))
    F = Ftip * Fhub

    # bem parameters
    k = cn*sigma_p/(4.0*F*sphi*sphi)
    kp = ct*sigma_p/(4.0*F*sphi*cphi)

    k2 = cn*sigma_p/(4.0*F*cphi*cphi)
    kp2 = ct*sigma_p/(4.0*F*sphi*sphi)

    return k, kp, F, k2, kp2

end



function residual(phi::Float64, r::Float64, chord::Float64, twist::Float64,
    Rhub::Float64, Rtip::Float64, af::AirfoilData, B::Int64,
    Vx::Float64, Vy::Float64)

    k, kp, F, _, _ = residualbase(phi, r, chord, twist, Rhub, Rtip, af, B, Vx, Vy)

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

    return R
end


# if isapprox(Vx, 0.0, rtol=1e-6)
function residualVx0(phi, r, chord, twist, Rhub, Rtip, af, B, Vx, Vy)

    _, _, _, k2, _ = residualbase(phi, r, chord, twist, Rhub, Rtip, af, B, Vx, Vy)

    s = sign(phi)
    u = s*sqrt(s*k2)*abs(Vy)

    R = - sin(phi)/u - cos(phi)/Vy

    return R
end

# elseif isapprox(Vy, 0.0, rtol=1e-6)
function residualVy0(phi, r, chord, twist, Rhub, Rtip, af, B, Vx, Vy)

    _, _, _, _, kp2 = residualbase(phi, r, chord, twist, Rhub, Rtip, af, B, Vx, Vy)

    v = kp2*abs(Vx)

    R = sin(phi)/Vx - cos(phi)/v

    return R
end
