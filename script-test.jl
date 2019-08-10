using CCBlade
using PyPlot

Rhub = 1.5
Rtip = 63.0
B = 3  # number of blades
turbine = true
precone = 2.5*pi/180

r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
    28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
    56.1667, 58.9000, 61.6333]
chord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
    3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
theta = pi/180*[13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
    6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]

aftypes = Array{Any}(undef, 8)
aftypes[1] = af_from_file("airfoils/Cylinder1.dat")
aftypes[2] = af_from_file("airfoils/Cylinder2.dat")
aftypes[3] = af_from_file("airfoils/DU40_A17.dat")
aftypes[4] = af_from_file("airfoils/DU35_A17.dat")
aftypes[5] = af_from_file("airfoils/DU30_A17.dat")
aftypes[6] = af_from_file("airfoils/DU25_A17.dat")
aftypes[7] = af_from_file("airfoils/DU21_A17.dat")
aftypes[8] = af_from_file("airfoils/NACA64_A17.dat")

af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

sections = Section.(r, chord, theta, aftypes[af_idx])

yaw = 0.0*pi/180
tilt = 5.0*pi/180
hubHt = 90.0
shearExp = 0.2

# operating point for the turbine/propeller
Vinf = 10.0
tsr = 7.55
rotorR = Rtip*cos(precone)
Omega = Vinf*tsr/rotorR
azimuth = 0.0*pi/180
rho = 1.225

inflows = windturbineinflow.(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

outputs = solve.(rotor, sections, inflows)

Np, Tp = loads(outputs)

# plot distributed loads
figure()
plot(r/Rtip, Np/1e3)
plot(r/Rtip, Tp/1e3)
xlabel("r/Rtip")
ylabel("distributed loads (kN/m)")
legend(["flapwise", "lead-lag"])
savefig("loads-turbine.svg"); nothing # hide

T, Q = thrusttorque(rotor, sections, outputs)

azangles = pi/180*[0.0, 90.0, 180.0, 270.0]
azinflows = windturbineinflow_az(Vinf, Omega, r, precone, yaw, tilt, azangles, hubHt, shearExp, rho)
T, Q = thrusttorque_azavg(rotor, sections, azinflows)

CP, CT, CQ = nondim(T, Q, Vinf, Omega, rho, rotor)

ntsr = 20  # number of tip-speed ratios
tsrvec = range(2, 15, length=ntsr)
cpvec = zeros(ntsr)  # initialize arrays
ctvec = zeros(ntsr)

azangles = pi/180*[0.0, 90.0, 180.0, 270.0]

for i = 1:ntsr
    Omega = Vinf*tsrvec[i]/rotorR

    azinflows = windturbineinflow_az(Vinf, Omega, r, precone, yaw, tilt, azangles, hubHt, shearExp, rho)
    T, Q = thrusttorque_azavg(rotor, sections, azinflows)

    cpvec[i], ctvec[i], _ = nondim(T, Q, Vinf, Omega, rho, rotor)
end

figure()
plot(tsrvec, cpvec)
plot(tsrvec, ctvec)
xlabel("tip speed ratio")
legend([L"C_P", L"C_T"])