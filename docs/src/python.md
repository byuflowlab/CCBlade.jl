# Calling CCBlade.jl from Python (optionally with derivatives)

In this example we repeat the [quick start](tutorial.md), but with Python.  We won't repeat all the usage details described in the quick start.  We are using the [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/) packages to enable this.  Notice that the main change from the Julia example is that we need to explicitly use the Julia broadcast functions (which is what the dot syntax does in Julia).

```python
import numpy as np
from juliacall import Main as jl
import matplotlib.pyplot as plt

jl.seval("using CCBlade")

Rtip = 10/2.0 * 0.0254  # inches to meters
Rhub = 0.10*Rtip
B = 2  # number of blades

rotor = jl.Rotor(Rhub, Rtip, B)

propgeom = np.array([
    [0.15, 0.130, 32.76],
    [0.20, 0.149, 37.19],
    [0.25, 0.173, 33.54],
    [0.30, 0.189, 29.25],
    [0.35, 0.197, 25.64],
    [0.40, 0.201, 22.54],
    [0.45, 0.200, 20.27],
    [0.50, 0.194, 18.46],
    [0.55, 0.186, 17.05],
    [0.60, 0.174, 15.97],
    [0.65, 0.160, 14.87],
    [0.70, 0.145, 14.09],
    [0.75, 0.128, 13.39],
    [0.80, 0.112, 12.84],
    [0.85, 0.096, 12.25],
    [0.90, 0.081, 11.37],
    [0.95, 0.061, 10.19],
    [1.00, 0.041, 8.99]]
)

r = propgeom[:, 0] * Rtip
chord = propgeom[:, 1] * Rtip
theta = propgeom[:, 2] * np.pi/180

af = jl.AlphaAF("data/naca4412.dat")

sections = jl.broadcast(jl.Section, r, chord, theta, jl.Ref(af))

Vinf = 5.0
Omega = 5400*np.pi/30  # convert to rad/s
rho = 1.225

op = jl.broadcast(jl.simple_op, Vinf, Omega, r, rho)

out = jl.broadcast(jl.solve, jl.Ref(rotor), sections, op)

plt.figure()
plt.plot(r/Rtip, out.Np)
plt.plot(r/Rtip, out.Tp)
plt.xlabel("r/Rtip")
plt.ylabel("distributed loads (N/m)")
plt.legend(["flapwise", "lead-lag"])
plt.show()
```

We now continue the example demonstrating how to get derivatives for use in Python (but where the derivative computationa occurs in Julia via algorithmic differentiation).  For this functionality we need to load the [PythonCall](https://juliapy.github.io/PythonCall.jl/stable/) package (which enables us to call back into Python from Julia), and we need the [ImplicitAD](https://github.com/byuflowlab/ImplicitAD.jl) packages which provides a convenience function for the derivative computation.  Alternatively, for more advanced users you can just use the Julia differentiation packages directly (ForwardDiff, ReverseDiff, etc.) as demonstrated in the [how to guide](howto.md).  Note that for Julia AD to work, all the function calls will need to be Julia function calls.  Even though all the setup is happening here in Python, we are only setting up inputs.  All functions are calls to Julia (jl.somefunction())

Not counting the one-time loading cost of starting up the Julia runtime, we find the computational time of computing derivatives to be essentially identical to computing in pure Julia (in other words there is minimal overhead between Python/Julia for these examples).  The below example demonstrates a forward mode Jacobian and a forward mode Jacobian-vector product.  Other options (reverse Jacobian, reverse vector-Jacobian product) are discussed in [ImplicitAD](https://github.com/byuflowlab/ImplicitAD.jl) docs.

```python
jl.seval("using PythonCall")
jl.seval("using ImplicitAD: derivativesetup")

nc = len(chord)

# ImplicitAD expects a funciton we want to differentiate in the form f = func(x, p)
# where f is output vector, x is input vector, and p are parameters we do not differentiate w.r.t.
def ccbladewrap(x, p):
    chord = x[:nc]
    theta = x[nc:]
    sections = jl.broadcast(jl.Section, r, chord, theta, jl.Ref(af))
    out = jl.broadcast(jl.solve, jl.Ref(rotor), sections, op)
    T, Q = jl.thrusttorque(rotor, sections, out)
    return [T, Q]

x = np.concatenate([chord, theta])
p = ()
jacobian = jl.derivativesetup(ccbladewrap, x, p, "fjacobian")  # a forward-mode Jacobian is one option

# preallocate Jacobian then evaluate
J = np.zeros((2, len(x)))
jacobian(J, x)
print(J)
# can now change x, and evaluate jacobian(J, x) repeatedly at other points


# demonstrate a Jacobian-vector product
jvp = jl.derivativesetup(ccbladewrap, x, p, "jvp")
xdot = np.ones(len(x))
fdot = np.zeros(2)
jvp(fdot, x, xdot)
print(fdot)
# can continue to call jvp for different x, xdot pairs
```