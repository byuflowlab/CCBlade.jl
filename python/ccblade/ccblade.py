import numpy as np
from juliacall import Main as jl

# Include the Julia package
jl.seval("using Pkg")
# jl.seval(f'Pkg.activate("{julia_project_path}")')
jl.seval("using CCBlade")

Rotor = jl.Rotor
AlphaAF = jl.AlphaAF
broadcast= jl.broadcast
Section = jl.Section
Ref = jl.Ref
simple_op = jl.simple_op
solve  = jl.solve



