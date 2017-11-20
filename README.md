# CCBlade.jl

Blade Element Momentum Method for Propellers and Turbines.  
Author: Andrew Ning

Originally based on my paper: http://dx.doi.org/10.1002/we.1636 and code: https://github.com/wisdem/ccblade.  But extended for propellers, wind reversals, and no wind in one direction.  See theory doc for details.

### (internal) Installation
Clone this repo in to a folder of local julia packages.  Then add the following to your .juliarc.jl file.
```julia
push!(LOAD_PATH, "/path/to/your/julia-local-packages")
```
Then use as normal: 
```julia
using CCBlade
```

