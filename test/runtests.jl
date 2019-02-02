using CCBlade
using Test


# --- verification using example from "Wind Turbine Blade Analysis using the Blade Element Momentum Method"
# --- by Grant Ingram: https://community.dur.ac.uk/g.l.ingram/download/wind_turbine_design.pdf

# Note: There were various problems with the pdf data. Fortunately the excel
# spreadsheet was provided: http://community.dur.ac.uk/g.l.ingram/download.php

# - They didn't actually use the NACA0012 data.  According to the spreadsheet they just used cl = 0.084*alpha with alpha in degrees.
# - There is an error in the spreadsheet where CL at the root is always 0.7.  This is because the classical approach doesn't converge properly at that station.
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

outputs = solve.(sections, inflows, rotor)

# Np = getfield.(outputs, :Np)
Np = [o.Np for o in outputs]
avec = [o.u/Vinf for o in outputs]
apvec = [o.v for o in outputs] ./ (Omega*r)
phivec = [o.phi for o in outputs]

ivec = phivec*180/pi .- theta*180/pi
betavec = 90 .- phivec*180/pi


# outputs[1] is uncomparable because the classical method fails at the root so they fixed cl

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


# -------- verification: propellers.  using script at http://www.aerodynamics4students.com/propulsion/blade-element-propeller-theory.php ------
# I increased their tolerance to 1e-6

# inputs
chord = 0.10
D = 1.6
RPM = 2100
rho = 1.225
pitch = 1.0  # pitch distance in meters.


# --- rotor definition ---
turbine = false
Rhub = 0.01
Rtip = 100.0  # something large to eliminate tip effects
B = 2  # number of blades

rotor = Rotor(Rhub, Rtip, B, turbine)

# --- section definitions ---

R = D/2.0
r = range(R/10, stop=R, length=11)
theta = atan.(pitch./(2*pi*r))


function affunc(alpha, Re, M)

    cl = 6.2*alpha
    cd = 0.008 - 0.003*cl + 0.01*cl*cl

    return cl, cd
end 

sections = Section.(r, chord, theta, affunc)


# --- inflow definitions ---

tsim = 1e3*[1.045361193032356, 1.025630300048415, 1.005234466788998, 0.984163367036026, 0.962407923825140, 0.939960208707079, 0.916813564966455, 0.892962691000145, 0.868403981825492, 0.843134981103815, 0.817154838249790, 0.790463442573673, 0.763063053839278, 0.734956576558370, 0.706148261507327, 0.676643975451150, 0.646450304160057, 0.615575090105131, 0.584027074365864, 0.551815917391907, 0.518952127358381, 0.485446691671386, 0.451311288662196, 0.416557935286392, 0.381199277009438, 0.345247916141561, 0.308716772800348, 0.271618894441869, 0.233967425339051, 0.195775319296371, 0.157055230270717, 0.117820154495231, 0.078082266879117, 0.037854059197644, -0.002852754149850, -0.044026182837742, -0.085655305814570, -0.127728999394140, -0.170237722799272, -0.213169213043848, -0.256515079286031, -0.300266519551194, -0.344414094748869, -0.388949215983616, -0.433863576642539, -0.479150401337354, -0.524801553114807, -0.570810405128802, -0.617169893200684, -0.663873474163182, -0.710915862524620, -0.758291877949762, -0.805995685105502, -0.854022273120508, -0.902366919041604, -0.951025170820984, -0.999992624287163, -1.049265666456123, -1.098840222937414, -1.148712509929845]
qsim = 1e2*[0.803638686218187, 0.806984572453978, 0.809709290183008, 0.811743686838315, 0.813015017103876, 0.813446921530685, 0.812959654049620, 0.811470393912576, 0.808893852696513, 0.805141916379142, 0.800124489784850, 0.793748780791057, 0.785921727832179, 0.776548246109426, 0.765532528164390, 0.752778882688809, 0.738190986274448, 0.721673076180745, 0.703129918771009, 0.682467282681955, 0.659592296506578, 0.634413303042323, 0.606840565246423, 0.576786093366321, 0.544164450503912, 0.508891967461804, 0.470887571011192, 0.430072787279711, 0.386371788290446, 0.339711042057184, 0.290019539402947, 0.237229503458026, 0.181274942660876, 0.122093307308376, 0.059623821454727, -0.006190834182631, -0.075406684829235, -0.148076528546541, -0.224253047813501, -0.303980950928302, -0.387309291734422, -0.474283793689904, -0.564946107631716, -0.659336973911858, -0.757495165410553, -0.859460291551374, -0.965266648683888, -1.074949504731187, -1.188540970723477, -1.306072104649531, -1.427575034895290, -1.553080300508925, -1.682614871422754, -1.816205997296014, -1.953879956474228, -2.095662107769925, -2.241576439746701, -2.391647474158875, -2.545897099743367, -2.704346566395035]

for i = 1:60

  Vinf = float(i)
  Omega = RPM * pi/30

  inflows = simpleinflow.(Vinf, Omega, r, rho)


  # --- evaluate ---

  outputs = solve.(sections, inflows, rotor)

  Np, Tp = loads(outputs)

  # T, Q = thrusttorque(r[1], r, r[end], Np, Tp, B)
  # their spreadsheet did not use trapzezoidal rule, so just do a rect sum.
  T = sum(Np*(r[2]-r[1]))*B
  Q = sum(r.*Tp*(r[2]-r[1]))*B

  @test isapprox(T, tsim[i], atol=1e-2)  # 2 decimal places
  @test isapprox(Q, qsim[i], atol=2e-3)

end


# 
# 

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



# # --- rotor definition ---
# turbine = false
# Rhub =.0254*.5
# Rtip = .0254*3.0
# B = 2  # number of blades

# rotor = Rotor(Rhub, Rtip, B, turbine)


# # --- section definitions ---

# r = .0254*[0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332,
#      1.1128, 1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911,
#      1.6708, 1.7505, 1.8302, 1.9099, 1.9896, 2.0693, 2.1490, 2.2287,
#      2.3084, 2.3881, 2.4678, 2.5475, 2.6273, 2.7070, 2.7867, 2.8661, 2.9410]
# chord = .0254*[0.6270, 0.6255, 0.6231, 0.6199, 0.6165, 0.6125, 0.6054, 0.5973, 0.5887,
#           0.5794, 0.5695, 0.5590, 0.5479, 0.5362, 0.5240, 0.5111, 0.4977,
#           0.4836, 0.4689, 0.4537, 0.4379, 0.4214, 0.4044, 0.3867, 0.3685,
#           0.3497, 0.3303, 0.3103, 0.2897, 0.2618, 0.1920]
# theta = pi/180.0*[40.2273, 38.7657, 37.3913, 36.0981, 34.8803, 33.5899, 31.6400,
#                    29.7730, 28.0952, 26.5833, 25.2155, 23.9736, 22.8421, 21.8075,
#                    20.8586, 19.9855, 19.1800, 18.4347, 17.7434, 17.1005, 16.5013,
#                    15.9417, 15.4179, 14.9266, 14.4650, 14.0306, 13.6210, 13.2343,
#                    12.8685, 12.5233, 12.2138]

# affunc2 = af_from_file("airfoils/NACA64_A17.dat")

# sections = Section.(r, chord, theta, affunc2)


# # --- inflow definitions ---

# Vinf = 10.0
# Omega = 8000.0*pi/30.0
# rho = 1.225

# inflows = simpleinflow.(Vinf, Omega, r, rho)


# # --- evaluate ---

# outputs = solve.(sections, inflows, rotor)
# Np, Tp = loads(outputs)

# using PyPlot
# figure()
# plot(r/Rtip, Np)
# plot(r/Rtip, Tp)





# J = linspace(0.1, 0.9, 20)

# Omega = 8000.0*pi/30
# n = Omega/(2*pi)
# D = 2*Rtip*cos(precone)

# eff = zeros(20)
# CT = zeros(20)
# CQ = zeros(20)

# for i = 1:20
#     Vinf = J[i] * D * n

#     inflow = simpleInflow(Vinf, Omega, r, precone, rho)

#     T, Q = thrusttorque(rotor, [inflow], turbine)
#     eff[i], CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, Rtip, precone, turbine)

# end


# figure()
# plot(J, CT)
# plot(J, CQ*2*pi)
# xlabel(L"J")
# legend([L"C_T", L"C_P"])