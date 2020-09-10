# Changelog


## [0.2] - Sep 2020

- small changes to airfoil file format for better use with Re/Mach variation
- new expandable airfoil types to allow variation in angle of attack, Reynolds number, Mach number
- integrated (and overloadable) airfoil corrections for both on the fly and preprocessing changes (Re, Mach, rotation)
- overloadable tip/hub loss corrections
- allow r to go all the way to Rhub and Rtip for convenience
- airfoil extrapolation function for convenience
- specific residuals to handle Vx = 0 (hover) and Vy = 0 (parked) inflow cases.
- (internal) code written in propeller conventions though provides both turbine and propeller conventions outwardly.
- new unit tests for airfoil corrections, derivatives, and type stability
- overhaul of documentation based on principles discussed [here](https://documentation.divio.com)


## [0.1] - May 2020

Initial registered version.