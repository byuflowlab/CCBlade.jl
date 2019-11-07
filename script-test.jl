using CCBlade
using PyPlot

Rhub = 1.5
Rtip = 63.0
B = 3  # number of blades
turbine = true
pitch = 0.0
precone = 2.5*pi/180

r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
    28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
    56.1667, 58.9000, 61.6333]
chord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
    3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
theta = pi/180*[13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
    6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]

aftypes = Array{Any}(undef, 8)
aftypes[1] = af_from_files("airfoils/Cylinder1.dat")
aftypes[2] = af_from_files("airfoils/Cylinder2.dat")
aftypes[3] = af_from_files("airfoils/DU40_A17.dat")
aftypes[4] = af_from_files("airfoils/DU35_A17.dat")
aftypes[5] = af_from_files("airfoils/DU30_A17.dat")
aftypes[6] = af_from_files("airfoils/DU25_A17.dat")
aftypes[7] = af_from_files("airfoils/DU21_A17.dat")
aftypes[8] = af_from_files("airfoils/NACA64_A17.dat")

af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

airfoils = aftypes[af_idx]


afexample = af_from_files(["airfoils/DU25_A17.dat", "airfoils/DU25_A17.dat"], Re=[1e5, 2e5])
cl, cd = afexample(.1, 1e5, 0)

afexample = af_from_files(["airfoils/DU25_A17.dat", "airfoils/DU25_A17.dat"], Mach=[0.6, 0.7])
cl, cd = afexample(.1, 0, 0.65)

afexample = af_from_files([["airfoils/DU25_A17.dat" "airfoils/DU25_A17.dat" "airfoils/DU25_A17.dat"]; 
    ["airfoils/DU25_A17.dat" "airfoils/DU25_A17.dat" "airfoils/DU25_A17.dat"]], 
    Re=[1e5, 2e5], Mach=[0.6, 0.68, 0.7])
cl, cd = afexample(.1, 1.8e5, 0.65)

rotor = Rotor(Rhub, Rtip, B, turbine, pitch, precone)
sections = Section.(r, chord, theta, airfoils)

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

op = windturbine_op.(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

out = solve.(Ref(rotor), sections, op)

# plot distributed loads
figure()
plot(r/Rtip, out.Np/1e3)
plot(r/Rtip, out.Tp/1e3)
xlabel("r/Rtip")
ylabel("distributed loads (kN/m)")
legend(["flapwise", "lead-lag"])
# savefig("loads-turbine.svg"); nothing # hide

T, Q = thrusttorque(rotor, sections, out)

azangles = pi/180*[0.0, 90.0, 180.0, 270.0]
ops = windturbine_op.(Vinf, Omega, r, precone, yaw, tilt, azangles', hubHt, shearExp, rho)
outs = solve.(Ref(rotor), sections, ops)

T, Q = thrusttorque(rotor, sections, outs)

CP, CT, CQ = nondim(T, Q, Vinf, Omega, rho, rotor)

ntsr = 20  # number of tip-speed ratios
tsrvec = range(2, 15, length=ntsr)
cpvec = zeros(ntsr)  # initialize arrays
ctvec = zeros(ntsr)

azangles = pi/180*[0.0, 90.0, 180.0, 270.0]

for i = 1:ntsr
    Omega = Vinf*tsrvec[i]/rotorR

    ops = windturbine_op.(Vinf, Omega, r, precone, yaw, tilt, azangles', hubHt, shearExp, rho)
    outs = solve.(Ref(rotor), sections, ops)
    T, Q = thrusttorque(rotor, sections, outs)

    cpvec[i], ctvec[i], _ = nondim(T, Q, Vinf, Omega, rho, rotor)
end

figure()
plot(tsrvec, cpvec)
plot(tsrvec, ctvec)
xlabel("tip speed ratio")
legend([L"C_P", L"C_T"])
# gcf()


Rhub = 0.0254*.5
Rtip = 0.0254*3.0
B = 2  # number of blades
turbine = false

# section definition
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

af = af_from_files("airfoils/NACA64_A17.dat")
airfoils = fill(af, length(r))

rotor = Rotor(Rhub, Rtip, B, turbine)
sections = Section.(r, chord, theta, airfoils)

rho = 1.225
Vinf = 10.0
Omega = 8000.0*pi/30.0

op = simple_op.(Vinf, Omega, r, rho)

outputs = solve.(Ref(rotor), sections, op)

figure()
plot(r/Rtip, outputs.Np)
plot(r/Rtip, outputs.Tp)
xlabel("r/Rtip")
ylabel("distributed loads (N/m)")
legend(["flapwise", "lead-lag"])
# gcf()

# u, v = effectivewake(outputs)

figure()
plot(r/Rtip, outputs.u/Vinf)
plot(r/Rtip, outputs.v/Vinf)
xlabel("r/Rtip")
ylabel("(normalized) induced velocity at rotor disk")
legend(["axial velocity", "swirl velocity"])
# gcf()

nJ = 20  # number of advance ratios


J = range(0.1, 0.9, length=nJ)  # advance ratio

Omega = 8000.0*pi/30
n = Omega/(2*pi)
D = 2*Rtip

eff = zeros(nJ)
CT = zeros(nJ)
CQ = zeros(nJ)

for i = 1:nJ
    Vinf = J[i] * D * n

    op = simple_op.(Vinf, Omega, r, rho)
    outputs = solve.(Ref(rotor), sections, op)
    T, Q = thrusttorque(rotor, sections, outputs)
    eff[i], CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, rotor)

end

figure()
plot(J, CT)
plot(J, CQ*2*pi)
xlabel(L"J")
legend([L"C_T", L"C_P"])
# gcf()
# savefig("ctcp-prop.svg") # hide

figure()
plot(J, eff)
xlabel(L"J")
ylabel(L"\eta")
# savefig("eta-prop.svg"); nothing # hide
# gcf()