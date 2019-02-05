# CCBlade Documentation

**Summary**: A blade element momentum method for propellers and turbines. 

**Author**: Andrew Ning

**Features**:

- Prandtl hub/tip losses
- Glauert/Buhl empirical region for high thrust turbines
- Convenience functions for inflow with shear, precone, yaw, tilt, and azimuth
- Allows for flow reversals (negative inflow/rotation velocities)
- Allows for a hover condition (only rotation, no inflow) and rotor locked (no rotation, only inflow)
- Methodology is provably convergent (see <http://dx.doi.org/10.1002/we.1636> although multiple improvements have been made since then)
- Compatible with AD tools like ForwardDiff

**Installation**:

```julia
pkg> add https://github.com/byuflowlab/CCBlade.jl
```

