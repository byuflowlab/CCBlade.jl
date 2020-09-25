# Changelog


## [0.2.1] - Sep 2020

- minor change to improve type flexibility in OperatingPoint (for AD usage)

## [0.2] - Sep 2020

- WARNING: default is now a propeller instead of a turbine.  But turbine conventions are easily accessed through a flag.  Main reason was to better support rotorcraft usage, and because we already have example implementations from a turbine perspective thought it would be nice for the internal representation to use a propeller perspective so both are available.
- small changes to airfoil file format for better use with Re/Mach variation
- new expandable airfoil types to allow variation in angle of attack, Reynolds number, Mach number
- integrated (and overloadable) airfoil corrections for both on the fly and preprocessing changes (Re, Mach, rotation)
- overloadable tip/hub loss corrections
- allow r to go all the way to Rhub and Rtip for convenience
- airfoil extrapolation function for convenience
- specific residuals to handle Vx = 0 (hover) and Vy = 0 (parked) inflow cases.
- overhaul of documentation based on principles discussed [here](https://documentation.divio.com)

## [0.1] - May 2020

Initial registered version.