# CCBlade.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://flow.byu.edu/CCBlade.jl/stable)
![](https://github.com/byuflowlab/CCBlade.jl/workflows/Run%20tests/badge.svg)


**Summary**: A blade element momentum method for propellers and turbines. 

**Author**: Andrew Ning

**Features**:

- Methodology is provably convergent (see <http://dx.doi.org/10.1002/we.1636> although multiple improvements have been made since then)
- Prandtl hub/tip losses (or user-defined losses)
- Glauert/Buhl empirical region for high thrust turbines
- Convenience functions for inflow with shear, precone, yaw, tilt, and azimuth
- Can do airfoil corrections beforehand or on the fly (Mach, Reynolds, rotation)
- Allows for flow reversals (negative inflow/rotation velocities)
- Allows for a hover condition (only rotation, no inflow) and rotor locked (no rotation, only inflow)
- Compatible with AD tools like ForwardDiff

**Installation**:

```julia
] add CCBlade
```

**Documentation**:

The [documentation](https://flow.byu.edu/CCBlade.jl/stable/) contains
- A quick start tutorial to learn basic usage,
- Guided examples to address specific or more advanced tasks,
- A reference describing the API,
- Theory in full detail.

**Run Unit Tests**:

```julia
pkg> activate .
pkg> test
```

**Citing**:

Ning, A., “Using Blade Element Momentum Methods with Gradient-Based Design Optimization,” Structural and Multidisciplinary Optimization, Vol. 64, No. 2, pp. 994–1014, May 2021. doi:10.1007/s00158-021-02883-6

**Python / OpenMDAO users**

In the `openmdao` folder there is a Python wrapper to this package to enable usage from [OpenMDAO](https://openmdao.org).  This wrapper was developed/maintained by Daniel Ingraham and Justin Gray at NASA Glenn.

In the `python` folder there is a Python wrapper developed by BYU FLOWLab
