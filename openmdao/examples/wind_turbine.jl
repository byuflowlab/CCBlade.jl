using CCBlade

# --- rotor definition ---
r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
    28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
    56.1667, 58.9000, 61.6333]
chord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
    3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
theta = pi/180*[13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
    6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]
Rhub = 1.5
Rtip = 63.0
B = 3  # number of blades
turbine = true
pitch = 0.0
precone = 2.5*pi/180
rotor = Rotor(Rhub, Rtip, B, turbine, pitch, precone)

# Define airfoils.  In this case we have 8 different airfoils that we load into
# an array. These airfoils are defined in files.
airfoil_fnames = ["airfoils/Cylinder1.dat", "airfoils/Cylinder2.dat",
                  "airfoils/DU40_A17.dat", "airfoils/DU35_A17.dat",
                  "airfoils/DU30_A17.dat", "airfoils/DU25_A17.dat",
                  "airfoils/DU21_A17.dat", "airfoils/NACA64_A17.dat"]
aftypes = [af_from_files([f]) for f in airfoil_fnames]

# indices correspond to which airfoil is used at which station
af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

# create airfoil array 
airfoils = aftypes[af_idx]

sections = Section.(r, chord, theta, airfoils)

# operating point for the turbine
yaw = 0.0*pi/180
tilt = 5.0*pi/180
hubHt = 90.0
shearExp = 0.2

Vinf = 10.0
tsr = 7.55
rotorR = Rtip*cos(precone)
Omega = Vinf*tsr/rotorR
azimuth = 0.0*pi/180
rho = 1.225

op = windturbine_op.(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)
@show op[1].Vx
@show op[1].Vy
@show sections[1].r
@show sections[1].chord
@show sections[1].theta
@show op[1].rho
@show rotor.Rhub
@show rotor.Rtip
@show rotor.B
@show rotor.turbine
@show rotor.pitch
@show rotor.precone

outputs = solve.(rotor, sections, op)

@show outputs[1].phi
