# -------- wind turbine example ----------------

# geometry
Rhub = 1.5
Rtip = 63.0

r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
    28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
    56.1667, 58.9000, 61.6333]
chord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
    3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
theta = [13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
    6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]*pi/180
B = 3  # number of blades

aftypes = Array(AirfoilData, 8)
aftypes[1] = readaerodyn("airfoils/Cylinder1.dat")
aftypes[2] = readaerodyn("airfoils/Cylinder2.dat")
aftypes[3] = readaerodyn("airfoils/DU40_A17.dat")
aftypes[4] = readaerodyn("airfoils/DU35_A17.dat")
aftypes[5] = readaerodyn("airfoils/DU30_A17.dat")
aftypes[6] = readaerodyn("airfoils/DU25_A17.dat")
aftypes[7] = readaerodyn("airfoils/DU21_A17.dat")
aftypes[8] = readaerodyn("airfoils/NACA64_A17.dat")

af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

n = length(r)
af = Array(AirfoilData, n)
for i = 1:n
    af[i] = aftypes[af_idx[i]]
end

precone = 2.5*pi/180
yaw = 0.0*pi/180
tilt = 5.0*pi/180
hubHt = 90.0
shearExp = 0.2

# operating point for the turbine/propeller
Vinf = 10.0
tsr = 7.55
# pitch = 0.0*pi/180
rotorR = Rtip*cos(precone)
Omega = Vinf*tsr/rotorR
azimuth = 0.0*pi/180
rho = 1.225

inflow = windTurbineInflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)
rotor = Rotor(r, chord, theta, af, Rhub, Rtip, B, precone)

turbine = true

Np, Tp = distributedLoads(rotor, inflow, turbine)

tsrvec = linspace(2, 15, 20)
cpvec = zeros(20)
nsectors = 4

for i = 1:20
    Omega = Vinf*tsrvec[i]/rotorR

    inflow = windTurbineInflowMultiple(nsectors, Vinf, Omega, r, precone, yaw, tilt, hubHt, shearExp, rho)

    T, Q = thrusttorque(rotor, inflow, turbine)
    cpvec[i], CT, CQ = nondim(T, Q, Vinf, Omega, rho, Rtip, precone, turbine)

end


# using Plots
# pyplot()
using PyPlot
close("all")

figure()
plot(r/Rtip, Np/1e3)
plot(r/Rtip, Tp/1e3)
# display(p2)

figure()
plot(tsrvec, cpvec)


# include("/Users/andrewning/Dropbox/BYU/repos/gradients/ad.jl")
#
# function wrapper(x)
#     r = x[1:n]
#     chord = x[n+1:2*n]
#     theta = x[2*n+1:3*n]
#
#     rotor = Rotor(r, chord, theta, af, Rhub, Rtip, B, precone)
#
#     # inflow = windTurbineInflowMultiple(nsectors, Vinf, Omega, r, precone, yaw, tilt, hubHt, shearExp, rho)
#     # T, Q = thrusttorque(rotor, inflow, turbine)
#     # return [T; Q]
#
#     inflow = windTurbineInflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)
#     Np, Tp = distributedLoads(rotor, inflow, turbine)
#     return [Np; Tp]
# end
#
# x = [r; chord; theta]
# f1, gfd = GradEval.centraldiff(wrapper, x)
# f2, gad = GradEval.fad(wrapper, x)
# gerror = GradEval.check(wrapper, x, gad)
#
# # println(gfd)
# # println(gad)
# println(maximum(abs(gerror)))


# -------- propeller example ----------------


# geometry
Rhub =.0254*.5
Rtip = .0254*3.0

r = .0254*[0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332,
     1.1128, 1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911,
     1.6708, 1.7505, 1.8302, 1.9099, 1.9896, 2.0693, 2.1490, 2.2287,
     2.3084, 2.3881, 2.4678, 2.5475, 2.6273, 2.7070, 2.7867, 2.8661, 2.9410]
chord = .0254*[0.6270, 0.6255, 0.6231, 0.6199, 0.6165, 0.6125, 0.6054, 0.5973, 0.5887,
          0.5794, 0.5695, 0.5590, 0.5479, 0.5362, 0.5240, 0.5111, 0.4977,
          0.4836, 0.4689, 0.4537, 0.4379, 0.4214, 0.4044, 0.3867, 0.3685,
          0.3497, 0.3303, 0.3103, 0.2897, 0.2618, 0.1920]

theta = pi/180.0*[40.2273, 38.7657, 37.3913, 36.0981, 34.8803, 33.5899, 31.6400,
                   29.7730, 28.0952, 26.5833, 25.2155, 23.9736, 22.8421, 21.8075,
                   20.8586, 19.9855, 19.1800, 18.4347, 17.7434, 17.1005, 16.5013,
                   15.9417, 15.4179, 14.9266, 14.4650, 14.0306, 13.6210, 13.2343,
                   12.8685, 12.5233, 12.2138]
B = 2  # number of blades

aftype = readaerodyn("airfoils/NACA64_A17.dat")

n = length(r)
af = Array(AirfoilData, n)
for i = 1:n
    af[i] = aftype
end

precone = 0.0

rho = 1.225

Vinf = 10.0
Omega = 8000.0*pi/30.0

inflow = simpleInflow(Vinf, Omega, r, precone, rho)
rotor = Rotor(r, chord, theta, af, Rhub, Rtip, B, precone)

turbine = false

Np, Tp = distributedLoads(rotor, inflow, turbine)

figure()
plot(r/Rtip, Np)
plot(r/Rtip, Tp)


J = linspace(0.1, 0.9, 20)

Omega = 8000.0*pi/30
n = Omega/(2*pi)
D = 2*Rtip*cos(precone)

eff = zeros(20)
CT = zeros(20)
CQ = zeros(20)

for i = 1:20
    Vinf = J[i] * D * n

    inflow = simpleInflow(Vinf, Omega, r, precone, rho)

    T, Q = thrusttorque(rotor, [inflow], turbine)
    eff[i], CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, Rtip, precone, turbine)

end


figure()
plot(J, CT)

figure()
plot(J, CQ)

figure()
plot(J, eff)
