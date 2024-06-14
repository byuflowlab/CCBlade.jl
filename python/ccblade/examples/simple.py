from ccblade.ccblade import Rotor, AlphaAF, broadcast, Section, Ref ,simple_op ,solve 
import numpy as np
Rtip = 10/2.0 * 0.0254  # inches to meters
Rhub = 0.10*Rtip
B = 2  # number of blades

rotor = Rotor(Rhub, Rtip, B)

propgeom = np.array([
[0.15,   0.130,   32.76],
[0.20,   0.149,   37.19],
[0.25,   0.173,   33.54],
[0.30,   0.189,   29.25],
[0.35,   0.197,   25.64],
[0.40,   0.201,   22.54],
[0.45,   0.200,   20.27],
[0.50,   0.194,   18.46],
[0.55,   0.186,   17.05],
[0.60,   0.174,   15.97],
[0.65,   0.160,   14.87],
[0.70,   0.145,   14.09],
[0.75,   0.128,   13.39],
[0.80,   0.112,   12.84],
[0.85,   0.096,   12.25],
[0.90,   0.081,   11.37],
[0.95,   0.061,   10.19],
[1.00,   0.041,   8.99]
])

r = propgeom[:, 0] * Rtip
chord = propgeom[:, 1] * Rtip
theta = propgeom[:, 2] * np.pi/180

af = AlphaAF("../../../data/naca4412.dat")

sections = broadcast(Section, r, chord, theta, Ref(af))


Vinf = 5.0
Omega = 5400* np.pi/30  # convert to rad/s
rho = 1.225
op = broadcast(simple_op, Vinf, Omega, r, rho)

out = broadcast(solve, Ref(rotor), sections, op)
print(out.Np)
print(out.Tp)
