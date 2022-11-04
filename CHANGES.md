# Changelog

## [0.2.4] - Nov 2022

- more flexible AD compatibility
- included ImplicitAD to speed up derivative computation (minor speed in this case because residual is just 1D, maybe 2x)
- fixed Reynolds number used in one of the doc examples

## [0.2.3] - Nov 2021

Minor change in how the empirical region is handled based on suggestion from Kenneth Lønbæk.  In effect it is Buhl(F = 1)*F.  This forces CT -> 0 as F -> 0 at the tip.

## [0.2.2] - Aug 2021

- Two bug fixes for AlphaReMachAF functions

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
