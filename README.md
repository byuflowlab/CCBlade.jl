# CCBlade.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://flow.byu.edu/CCBlade.jl)


Blade Element Momentum Method for Propellers and Turbines.  
Author: Andrew Ning

Originally based on my paper: http://dx.doi.org/10.1002/we.1636 and my earlier python code: https://github.com/wisdem/ccblade.  But extended for propellers, wind reversals, no wind in one direction, etc.  See documentation for usage.  A theory document is also available for developers.

### Install

```julia
pkg> add https://github.com/byuflowlab/CCBlade.jl
```

<!-- (Be sure you've setup your SSH keys first as noted [here](https://docs.julialang.org/en/latest/manual/packages/#man-initial-setup-1)) -->

### Run Unit Tests

```julia
pkg> activate .
pkg> test
```

### Get Started

Read the [Guide](https://flow.byu.edu/CCBlade.jl) for examples.

### Develop

See [theory manual](https://byu.box.com/s/ewaj1apa6e6lzku0hb4e30qbumiffmiu).
