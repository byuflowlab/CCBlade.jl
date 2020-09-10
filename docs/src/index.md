# CCBlade Documentation

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
pkg> add CCBlade
```

**Documentation**:

- Start with the [quick start tutorial](tutorial.md) to learn basic usage.
- More advanced or specific queries are addressed in the [guided examples](howto.md).
- Full details of the API are listed in [reference](reference.md).
- Full details of the theory are linked from the [theory](theory.md) page.

**Run Unit Tests**

```julia
pkg> activate .
pkg> test
```

**Citing**:

Ning, A., “Using Blade Element Momentum Methods with Gradient-Based Design Optimization,” Apr. 2020, (in review).


