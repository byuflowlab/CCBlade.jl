# Guide

This section contains two examples, one for a wind turbine and one for a propeller.  Each function in the module is introduced along the way.  Because the workflow for the propeller is essentially the same as that for the wind turbine (except the operating point), the propeller example does not repeat the same level of detail or comments.

To start, we import CCBlade as well as a plotting module.  I prefer to use `import` rather than `using` as it makes the namespace clear when multiple modules exist (except with plotting where it is obvious), but for the purposes of keeping this example concise I will use `using`.

All angles should be in radians.  The only exception is the airfoil input file, which has angle of attack in degrees for readability.

```@example wt
using CCBlade
using PyPlot
```

## Wind Turbine

First, we need to define the geometric parameterization.  

```@docs
Rotor
```

Note that ``r`` is the distance along the blade, rather than in the rotor plane.  Rhub/Rtip define the hub and tip radius and are used for hub/tip corrections, for integration of loads, and for nondimensionalization.    The parameter `turbine` just changes the input/output conventions.  If `turbine=true` then the following positive directions for inputs, induced velocities, and loads are used as shown in the figure below.  

![](turbine.png)

The definition for the precone is shown below.  Note that there is a convenience constructor where precone is omitted (defaults to zero) because while precone is often used for wind turbines it is rarely used for propellers.

![](precone.png)

Let's define the rotor, except the airfoils which require a bit more explanation.  This example corresponds to the NREL 5MW reference wind turbine.

```@example wt

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
precone = 2.5*pi/180

nothing # hide
```

!!! warning
    You must have Rhub < r < Rtip for all r.  These are strict inequalities (not equal to).



The airfoil input is any function of the form: `cl, cd = airfoil(alpha, Re, Mach)` where `cl` and `cd` are the lift and drag coefficient respectively, `alpha` is the angle of attack in radians, `Re` is the Reynolds number, and `Mach` is the Mach number.  Some of the inputs can be ignored if desired (e.g., Mach number).  While users can supply any function, two convenience methods exist for creating these functions.  One takes data from a file, the other uses input arrays.  Both convert the data into a smooth cubic spline, with some smoothing to prevent spurious jumps, as a function of angle of attack (Re and Mach currently ignored).  

The required file format contains one header line that is ignored in parsing.  Its purpose is to allow you to record any information you want about the data in that file (e.g., where it came from, type of corrections applied, Reynolds number, etc.).  The rest of the file contains data in columns split by  whitespace (not commas) in the following order: alpha, cl, cd.  You can add additional columns of data (e.g., cm), but they will be ignored in this module.  Currently, only one Reynolds number is allowed.

For example, a simple file (a cylinder section) would look like:
```
Cylinder section with a Cd of 0.50.  Re = 1 million.
-180.0   0.000   0.5000   0.000
0.00     0.000   0.5000   0.000
180.0    0.000   0.5000   0.000
```

The function call is given by:

```@docs
af_from_file(filename)
```
Alternatively, if you have the alpha, cl, cd data already in vectors you can initialize directly from the vectors:

```@docs
af_from_data(alpha, cl, cd)
```

In this example, we will initialize from files since the data arrays would be rather long.  The only complication is that there are 8 different airfoils used at the 17 different radial stations so we need to assign them to the correct stations corresponding to the vector ``r`` defined previously.

```@example wt
# Define airfoils.  In this case we have 8 different airfoils that we load into an array.
# These airfoils are defined in files.
aftypes = Array{Any}(undef, 8)
aftypes[1] = af_from_file("airfoils/Cylinder1.dat")
aftypes[2] = af_from_file("airfoils/Cylinder2.dat")
aftypes[3] = af_from_file("airfoils/DU40_A17.dat")
aftypes[4] = af_from_file("airfoils/DU35_A17.dat")
aftypes[5] = af_from_file("airfoils/DU30_A17.dat")
aftypes[6] = af_from_file("airfoils/DU25_A17.dat")
aftypes[7] = af_from_file("airfoils/DU21_A17.dat")
aftypes[8] = af_from_file("airfoils/NACA64_A17.dat")

# indices correspond to which airfoil is used at which station
af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

# create airfoil array 
airfoils = aftypes[af_idx]

nothing # hide
```

We can now define the rotor geometry.

```@example wt
rotor = Rotor(r, chord, theta, airfoils, Rhub, Rtip, B, turbine, precone)
nothing # hide
```

Next, we need to specify the operating conditions.  At a basic level the inflow conditions need to be defined as a struct defined by OperatingPoint.  The parameters `mu` and `asound` are optional if Reynolds number and Mach number respectively are used in the airfoil functions.

```@docs
OperatingPoint
```

The coordinate system for Vx and Vy is shown at the top of [Wind Turbine](@ref).  In general, different inflow conditions will exist at every location along the blade, which is why Vx and Vy are arrays that should correspond to the radial `r` locations. The above type allows one to specify an arbitrary input definition, however, convenience methods exist for a few typical inflow conditions.  For a typical wind turbine operating point you can use the `windturbine_op_` function, which is based on the angles and equations shown below.

![](angles.png)

To account for the velocity change across the hub face we compute the height of each blade location relative to the hub using coordinate transformations (where ``\Phi`` is the precone angle):
```math
  z_h = r \cos\Phi \cos\psi \cos\Theta + r \sin\Phi\sin\Theta
```
then apply the shear exponent (``\alpha``):
```math
  V_{shear} = V_{hub} \left(1 + \frac{z_h}{H_{hub}} \right)^\alpha
```
where ``H_{hub}`` is the hub height.  Finally, we can compute the x- and y-components of velocity with additional coordinate transformations:
```math
\begin{aligned}
V_x &= V_{shear} ((\cos \gamma \sin \Theta \cos \psi + \sin \gamma \sin \psi)\sin \Phi + \cos \gamma \cos \Theta \cos \Phi)\\
V_y &= V_{shear} (\cos \gamma \sin \Theta\sin \psi - \sin \gamma \cos \psi) + \Omega r \cos\Phi
\end{aligned}
```


```@docs
windturbine_op
```

We will use this function for this example, at a tip-speed ratio of 7.55.  

```@example wt

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
pitch = 0.0

op = windturbine_op(Vinf, Omega, pitch, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)

nothing # hide
```

We have now defined the requisite inputs and can start using the BEM methodology.  The solve function is the core of the BEM.

```@docs
solve
```

The output result is a struct defined below. The positive directions for the normal and tangential forces, and the induced velocities are shown at the top of [Wind Turbine](@ref).

```@docs
Outputs
```

Note that we use broadcasting to solve all sections in one call.  

```@example wt
out = solve(rotor, op)

nothing # hide
```

Let's plot the distributed loads.


```@example wt
# plot distributed loads
figure()
plot(r/Rtip, out.Np/1e3)
plot(r/Rtip, out.Tp/1e3)
xlabel("r/Rtip")
ylabel("distributed loads (kN/m)")
legend(["flapwise", "lead-lag"])
savefig("loads-turbine.svg"); nothing # hide
```

![](loads-turbine.svg)


We are likely also interested in integrated loads, like thrust and torque, which are provided by the function `thrusttorque`.

```@docs
thrusttorque
```

```@example wt
T, Q = thrusttorque(rotor, out)
```

As used in the above example, this would give the thrust and torque assuming the inflow conditions were constant with azimuth (overly optimistic with this case at azimuth=0).  If one wanted to compute thrust and torque using azimuthal averaging you would compute multiple inflow conditions with different azimuth angles and then average the resulting forces.  This can be conveniently done with broadcasting.  We don't want to broadcast across `r` (we are broadcasting across the four different azimuth angles) so we need to wrap `r` in a `Ref()` statement.  Similarly, we don't want to broadcast the solve across rotor objects.  If uncomforable with broadcasting, this could all be done easily with a for loop.  The `thrusttorque` function is overloaded with a version that accepts an array of outputs rather than a single output and performs an integration using averaging across the conditions.


```@example wt
azangles = pi/180*[0.0, 90.0, 180.0, 270.0]
ops = windturbine_op.(Vinf, Omega, pitch, Ref(r), precone, yaw, tilt, azangles, hubHt, shearExp, rho)
outs = solve.(Ref(rotor), ops)

T, Q = thrusttorque(rotor, outs)
```

One final convenience function is to nondimensionalize the outputs.  The nondimensionalization uses different conventions depending on the application.  For a wind turbine the nondimensionalization is:

```math
\begin{aligned}
C_T &= \frac{T}{q A}\\
C_Q &= \frac{Q}{q R_{disk} A}\\
C_P &= \frac{P}{q A V_{hub}}\\
\end{aligned}
```

where
```math
\begin{aligned}
R_{disk} &= R_{tip} \cos(\text{precone})\\
A &= \pi R_{disk}^2\\
q &= \frac{1}{2}\rho V_{hub}^2\\
\end{aligned}
```

```@docs
nondim
```

```@example wt
CP, CT, CQ = nondim(T, Q, Vinf, Omega, rho, rotor)

```

As a final example, let's create a nondimensional power curve for this turbine (power coefficient vs tip-speed-ratio):

```@example wt
ntsr = 20  # number of tip-speed ratios
tsrvec = range(2, 15, length=ntsr)
cpvec = zeros(ntsr)  # initialize arrays
ctvec = zeros(ntsr)

azangles = pi/180*[0.0, 90.0, 180.0, 270.0]

for i = 1:ntsr
    Omega = Vinf*tsrvec[i]/rotorR

    ops = windturbine_op.(Vinf, Omega, pitch, Ref(r), precone, yaw, tilt, azangles, hubHt, shearExp, rho)
    outs = solve.(Ref(rotor), ops)
    T, Q = thrusttorque(rotor, outs)

    cpvec[i], ctvec[i], _ = nondim(T, Q, Vinf, Omega, rho, rotor)
end

figure()
plot(tsrvec, cpvec)
plot(tsrvec, ctvec)
xlabel("tip speed ratio")
legend([L"C_P", L"C_T"])
savefig("cpct-turbine.svg"); nothing # hide
```

![](cpct-turbine.svg)


## Propellers

The propeller analysis follows a very similar format.  The areas that are in common will not be repeated, only differences will be highlighted.

```@setup prop
using CCBlade
using PyPlot
```

Again, we first define the geometry, including the airfoils (which are the same along the blade in this case).  The positive conventions for a propeller (`turbine=false`) are shown in the figure below.  The underlying theory is unified across the two methods, but the input/output conventions differ to match common usage in the respective domains.

![](propeller.png)


```@example prop

# rotor definition
Rhub = 0.0254*.5
Rtip = 0.0254*3.0
B = 2  # number of blades
turbine = false

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

af = af_from_file("airfoils/NACA64_A17.dat")
airfoils = fill(af, length(r))

rotor = Rotor(r, chord, theta, airfoils, Rhub, Rtip, B, turbine)
nothing # hide
```

Next, we define the operating point.  For a propeller, it typically doesn't operate with tilt, yaw, and shear like a wind turbine does, so we have defined another convenience function for simple uniform inflow.  Like before, you can always define your own arbitrary inflow object.

```@docs
simple_op
```

In this example, we assume simple inflow.  

```@example prop

rho = 1.225

Vinf = 10.0
Omega = 8000.0*pi/30.0

op = simple_op(Vinf, Omega, r, rho)
nothing # hide
```

We can now computed distributed loads and induced velocities.  

```@example prop
outputs = solve(rotor, op)

figure()
plot(r/Rtip, outputs.Np)
plot(r/Rtip, outputs.Tp)
xlabel("r/Rtip")
ylabel("distributed loads (N/m)")
legend(["flapwise", "lead-lag"])
savefig("loads-prop.svg"); nothing # hide
```

![](loads-prop.svg)


This time we will also look at the induced velocities.  This is usually not of interest for wind turbines, but for propellers can be useful to assess, for example, prop-on-wing interactions.

```@example prop
figure()
plot(r/Rtip, outputs.u/Vinf)
plot(r/Rtip, outputs.v/Vinf)
xlabel("r/Rtip")
ylabel("(normalized) induced velocity at rotor disk")
legend(["axial velocity", "swirl velocity"])
savefig("velocity-prop.svg"); nothing # hide
```

![](velocity-prop.svg)


As before, we'd like to evaluate integrated quantities at multiple conditions in a for loop (advance ratios as is convention for propellers instead of tip-speed ratios).  The normalization conventions for a propeller are:

```math
\begin{aligned}
C_T &= \frac{T}{\rho n^2 D^4}\\
C_Q &= \frac{Q}{\rho n^2 D^5}\\
C_P &= \frac{P}{\rho n^3 D^5} = \frac{C_Q}{2 \pi}\\
\eta &= \frac{C_T J}{C_P}
\end{aligned}
```
where
```math
\begin{aligned}
n &= \frac{\Omega}{2\pi} \text{ rev per sec}\\
D &= 2 R_{tip} \cos(\text{precone})\\
J &= \frac{V}{n D}
\end{aligned}
```

!!! note
    Efficiency is set to zero if the thrust is negative (producing drag).  

The code below performs this analysis then plots thrust coefficient, power coefficient, and efficiency as a function of advance ratio.


```@example prop
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

    op = simple_op(Vinf, Omega, r, rho)
    outputs = solve(rotor, op)
    T, Q = thrusttorque(rotor, outputs)
    eff[i], CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, rotor)

end

figure()
plot(J, CT)
plot(J, CQ*2*pi)
xlabel(L"J")
legend([L"C_T", L"C_P"])
savefig("ctcp-prop.svg") # hide

figure()
plot(J, eff)
xlabel(L"J")
ylabel(L"\eta")
savefig("eta-prop.svg"); nothing # hide
```

![](ctcp-prop.svg)
![](eta-prop.svg)