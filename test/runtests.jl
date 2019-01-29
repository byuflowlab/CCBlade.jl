using CCBlade
using Test


# --- verification using example from "Wind Turbine Blade Analysis using the Blade Element Momentum Method"
# --- by Grant Ingram: https://community.dur.ac.uk/g.l.ingram/download/wind_turbine_design.pdf

# Note: There were various problems with the pdf data. Fortunately the excel
# spreadsheet was provided: http://community.dur.ac.uk/g.l.ingram/download.php

# - They didn't actually use the NACA0012 data.  According to the spreadsheet they just used cl = 0.084*alpha with alpha in degrees.
# - There is an error in the spreadsheet where CL at the root is always 0.7.  This is because their simple method doesn't converge properly at that station.
# - the values for gamma and chord were rounded in the pdf.  The spreadsheet has more precise values.
# - the tip is not actually converged (see significant error for Delta A).  I converged it further using their method.


# --- rotor definition ---
turbine = true
Rhub = 0.01
Rtip = 5.0
Rtip_eff = 5.0*100  # to eliminate tip effects as consistent with their study.
B = 3  # number of blades

rotor = Rotor(Rhub, Rtip_eff, B, turbine)


# --- section definitions ---

r = [0.2, 1, 2, 3, 4, 5]
gamma = [61.0, 74.31002131, 84.89805553, 89.07195504, 91.25038415, 92.58003871]
theta = (90.0 .- gamma)*pi/180
chord = [0.7, 0.706025153, 0.436187551, 0.304517933, 0.232257636, 0.187279622]

function affunc(alpha, Re, M)

    cl = 0.084*alpha*180/pi

    return cl, 0.0
end 

sections = Section.(r, chord, theta, affunc)


# --- inflow definitions ---

Vinf = 7.0
tsr = 8
Omega = tsr*Vinf/Rtip
rho = 1.0

inflows = simpleinflow.(Vinf, Omega, r, rho)

# --- evaluate ---

outputs = distributedloads.(sections, inflows, rotor)

# Np = getfield.(outputs, :Np)
Np = [o.Np for o in outputs]
avec = [o.u/Vinf for o in outputs]
apvec = [o.v for o in outputs] ./ (Omega*r)
phivec = [o.phi for o in outputs]

ivec = phivec*180/pi .- theta*180/pi
betavec = 90 .- phivec*180/pi


# outputs[1] is uncomparable because their method fails at the root so they fixed cl

@test isapprox(avec[2], 0.2443, atol=1e-4)
@test isapprox(apvec[2], 0.0676, atol=1e-4)
@test isapprox(avec[3], 0.2497, atol=1e-4)
@test isapprox(apvec[3], 0.0180, atol=1e-4)
@test isapprox(avec[4], 0.2533, atol=1e-4)
@test isapprox(apvec[4], 0.0081, atol=1e-4)
@test isapprox(avec[5], 0.2556, atol=1e-4)
@test isapprox(apvec[5], 0.0046, atol=1e-4)
@test isapprox(avec[6], 0.25725, atol=1e-4)  # note that their spreadsheet is not converged so I ran their method longer.
@test isapprox(apvec[6], 0.0030, atol=1e-4)

@test isapprox(betavec[2], 66.1354, atol=1e-3)
@test isapprox(betavec[3], 77.0298, atol=1e-3)
@test isapprox(betavec[4], 81.2283, atol=1e-3)
@test isapprox(betavec[5], 83.3961, atol=1e-3)
@test isapprox(betavec[6], 84.7113, atol=1e-3)  # using my more converged solution



# idx = 6

# aguess = avec[idx]
# apguess = apvec[idx]

# # aguess = 0.2557
# # apguess = 0.0046

# for i = 1:100

#     sigmap = B*chord[idx]/(2*pi*r[idx])
#     lambdar = inflows[idx].Vy/inflows[idx].Vx
#     beta2 = atan(lambdar*(1 + apguess)/(1 - aguess))
#     inc2 = gamma[idx]*pi/180 - beta2
#     cl2, cd2 = affunc(inc2, 1.0, 1.0)
#     global aguess = 1.0/(1 + 4*cos(beta2)^2/(sigmap*cl2*sin(beta2)))
#     global apguess = sigmap*cl2/(4*lambdar*cos(beta2))*(1 - a2)
#     # global apguess = 1.0 / (4*sin(beta2)/(sigmap*cl2) - 1)

# end

# aguess
# apguess


# # --- rotor definition ---
# turbine = true
# Rhub = 1.5
# Rtip = 63.0
# B = 3  # number of blades

# rotor = Rotor(Rhub, Rtip, B, turbine)

# # --- section definitions ---

# r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
#     28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
#     56.1667, 58.9000, 61.6333]
# chord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
#     3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
# theta = [13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
#     6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]*pi/180

# aftypes = Array{Tuple}(undef, 8)
# aftypes[1] = af_from_file("airfoils/Cylinder1.dat")
# aftypes[2] = af_from_file("airfoils/Cylinder2.dat")
# aftypes[3] = af_from_file("airfoils/DU40_A17.dat")
# aftypes[4] = af_from_file("airfoils/DU35_A17.dat")
# aftypes[5] = af_from_file("airfoils/DU30_A17.dat")
# aftypes[6] = af_from_file("airfoils/DU25_A17.dat")
# aftypes[7] = af_from_file("airfoils/DU21_A17.dat")
# aftypes[8] = af_from_file("airfoils/NACA64_A17.dat")

# af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

# sections = Section.(r, chord, theta, aftypes[af_idx])


# # ---- inflow definition --------

# precone = 2.5*pi/180
# yaw = 0.0*pi/180
# tilt = 5.0*pi/180
# hubHt = 90.0
# shearExp = 0.2


# # operating point for the turbine/propeller
# Vinf = 10.0
# tsr = 7.55
# # pitch = 0.0*pi/180
# rotorR = Rtip*cos(precone)
# Omega = Vinf*tsr/rotorR
# azimuth = 0.0*pi/180
# rho = 1.225
# mu = 1.8e-5
# asound = 1.0

# inflows = windturbineinflow.(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho, mu, asound)

# # -----------------


# outputs = distributedloads.(sections, inflows, rotor)

# Np = [out.Np for out in outputs]
# Tp = [out.Tp for out in outputs]

# using PyPlot

# figure()
# plot(r/Rtip, Np/1e3)
# plot(r/Rtip, Tp/1e3)

# # tsrvec = linspace(2, 15, 20)
# # cpvec = zeros(20)
# # nsectors = 4

# # for i = 1:20
# #     Omega = Vinf*tsrvec[i]/rotorR

# #     inflow = windTurbineInflowMultiple(nsectors, Vinf, Omega, r, precone, yaw, tilt, hubHt, shearExp, rho)

# #     T, Q = thrusttorque(rotor, inflow, turbine)
# #     cpvec[i], CT, CQ = nondim(T, Q, Vinf, Omega, rho, Rtip, precone, turbine)

# # end


# # figure()
# # plot(r/Rtip, Np/1e3)
# # plot(r/Rtip, Tp/1e3)
# # # display(p2)

# # figure()
# # plot(tsrvec, cpvec)

