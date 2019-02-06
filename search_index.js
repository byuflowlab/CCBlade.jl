var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "CCBlade Documentation",
    "title": "CCBlade Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "#CCBlade-Documentation-1",
    "page": "CCBlade Documentation",
    "title": "CCBlade Documentation",
    "category": "section",
    "text": "Summary: A blade element momentum method for propellers and turbines. Author: Andrew NingFeatures:Prandtl hub/tip losses\nGlauert/Buhl empirical region for high thrust turbines\nConvenience functions for inflow with shear, precone, yaw, tilt, and azimuth\nAllows for flow reversals (negative inflow/rotation velocities)\nAllows for a hover condition (only rotation, no inflow) and rotor locked (no rotation, only inflow)\nMethodology is provably convergent (see http://dx.doi.org/10.1002/we.1636 although multiple improvements have been made since then)\nCompatible with AD tools like ForwardDiffInstallation:pkg> add https://github.com/byuflowlab/CCBlade.jl"
},

{
    "location": "tutorial/#",
    "page": "Guide",
    "title": "Guide",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Guide-1",
    "page": "Guide",
    "title": "Guide",
    "category": "section",
    "text": "This section contains two examples, one for a wind turbine and one for a propeller.  Each function in the module is introduced along the way.  Because the workflow for the propeller is essentially the same as that for the wind turbine (except the inflow), the propeller example does not repeat the same level of detail or comments.To start, we import CCBlade as well as a plotting module.  I prefer to use import rather than using as it makes the namespace clear when multiple modules exist (except with plotting where it is obvious), but for the purposes of keeping this example concise I will use using.All angles should be in radians.  The only exception is the airfoil input file, which has angle of attack in degrees for readability.using CCBlade\nusing PyPlot"
},

{
    "location": "tutorial/#CCBlade.Rotor",
    "page": "Guide",
    "title": "CCBlade.Rotor",
    "category": "type",
    "text": "Rotor(Rhub, Rtip, B, turbine, precone=0.0)\n\nDefine geometry common to the entire rotor.\n\nArguments\n\nRhub::Float64: hub radius (along blade length)\nRtip::Float64: tip radius (along blade length)\nB::Int64: number of blades\nturbine::Bool: true if turbine, false if propeller\nprecone::Float64: precone angle\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.Section",
    "page": "Guide",
    "title": "CCBlade.Section",
    "category": "type",
    "text": "Section(r, chord, theta, af)\n\nDefine section properties at a given radial location on the rotor\n\nArguments\n\nr::Float64: radial location (Rhub < r < Rtip)\nchord::Float64: local chord length\ntheta::Float64: twist angle (radians)\naf: a function of the form: cl, cd = af(alpha, Re, Mach)\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.af_from_file-Tuple{Any}",
    "page": "Guide",
    "title": "CCBlade.af_from_file",
    "category": "method",
    "text": "af_from_file(filename)\n\nRead an airfoil file. Currently only reads one Reynolds number. Additional data like cm is optional but will be ignored. alpha should be in degrees\n\nformat:\n\nheader\n\nalpha1 cl1 cd1\n\nalpha2 cl2 cd2\n\nalpha3 cl3 cd3\n\n...\n\nReturns a function of the form cl, cd = func(alpha, Re, M) although Re and M are currently ignored.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.af_from_data-Tuple{Any,Any,Any}",
    "page": "Guide",
    "title": "CCBlade.af_from_data",
    "category": "method",
    "text": "af_from_data(alpha, cl, cd)\n\nCreate an AirfoilData object directly from alpha, cl, and cd arrays. alpha should be in radians.\n\naf_from_file calls this function indirectly.  Uses a cubic B-spline (if the order of the data permits it).  A small amount of smoothing of lift and drag coefficients is also applied to aid performance for gradient-based optimization.\n\nReturns a function of the form cl, cd = func(alpha, Re, M) although Re and M are currently ignored.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.Inflow",
    "page": "Guide",
    "title": "CCBlade.Inflow",
    "category": "type",
    "text": "Inflow(Vx, Vy, rho, mu=1.0, asound=1.0)\n\nOperation point for a rotor.  The x direction is the axial direction, and y direction is the tangential direction in the rotor plane.  See Documentation for more detail on coordinate systems.\n\nArguments\n\nVx::Float64: velocity in x-direction\nVy::Float64: velocity in y-direction\nrho::Float64: fluid density\nmu::Float64: fluid dynamic viscosity (unused if Re not included in airfoil data)\nasound::Float64: fluid speed of sound (unused if Mach not included in airfoil data)\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.windturbineinflow",
    "page": "Guide",
    "title": "CCBlade.windturbineinflow",
    "category": "function",
    "text": "windturbineinflow(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)\n\nCompute relative wind velocity components along blade accounting for inflow conditions and orientation of turbine.  See Documentation for angle definitions.\n\nArguments\n\nVhub::Float64: freestream speed at hub (m/s)\nOmega::Float64: rotation speed (rad/s)\nr::Array{Float64, 1}: radial locations where inflow is computed (m)\nprecone::Float64: precone angle (rad)\nyaw::Float64: yaw angle (rad)\ntilt::Float64: tilt angle (rad)\nazimuth::Float64: azimuth angle (rad)\nhubHt::Float64: hub height (m) - used for shear\nshearExp::Float64: power law shear exponent\nrho::Float64: air density (kg/m^3)\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.solve",
    "page": "Guide",
    "title": "CCBlade.solve",
    "category": "function",
    "text": "solve(section, inflow, rotor)\n\nSolve the BEM equations for one section, with given inflow conditions, and rotor properties. If multiple sections are to be solved (typical usage) then one can use broadcasting: solve.(sections, inflows, rotor) where sections and inflows are arrays.\n\nArguments\n\nrotor::Rotor: rotor properties\nsection::Section: section properties\ninflow::Inflow: inflow conditions\n\nReturns\n\noutputs::Outputs: BEM output data including loads, induction factors, etc.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.Outputs",
    "page": "Guide",
    "title": "CCBlade.Outputs",
    "category": "type",
    "text": "Outputs(Np, Tp, a, ap, u, v, phi, W, cl, cd, F)\n\nOutputs from the BEM solver.\n\nArguments\n\nNp::Float64: normal force per unit length\nTp::Float64: tangential force per unit length\na::Float64: axial induction factor\nap::Float64: tangential induction factor\nu::Float64: axial induced velocity\nv::Float64: tangential induced velocity\nphi::Float64: inflow angle\nW::Float64: inflow velocity\ncl::Float64: lift coefficient\ncd::Float64: drag coefficient\nF::Float64: hub/tip loss correction\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.loads",
    "page": "Guide",
    "title": "CCBlade.loads",
    "category": "function",
    "text": "loads(outputs)\n\nExtract arrays for Np and Tp from the outputs.\n\nThis does not do any calculations.  It is simply a syntax shorthand as loads are usually what we mostly care about from the outputs.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.thrusttorque",
    "page": "Guide",
    "title": "CCBlade.thrusttorque",
    "category": "function",
    "text": "thrusttorque(rotor, sections, outputs)\n\nintegrate the thrust/torque across the blade,  including 0 loads at hub/tip, using a trapezoidal rule.\n\nArguments\n\nrotor::Rotor: rotor object\nsections::Array{Section, 1}: section data along blade\noutputs::Array{Outputs, 1}: output data along blade\n\nReturns\n\nT::Float64: thrust (along x-dir see Documentation)\nQ::Float64: torque (along x-dir see Documentation)\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.nondim",
    "page": "Guide",
    "title": "CCBlade.nondim",
    "category": "function",
    "text": "nondim(T, Q, Vhub, Omega, rho, rotor)\n\nNondimensionalize the outputs.\n\nArguments\n\nT::Float64: thrust (N)\nQ::Float64: torque (N-m)\nVhub::Float64: hub speed used in turbine normalization (m/s)\nOmega::Float64: rotation speed used in propeller normalization (rad/s)\nrho::Float64: air density (kg/m^3)\nrotor::Rotor: rotor object\n\nReturns\n\nif windturbine\n\nCP::Float64: power coefficient\nCT::Float64: thrust coefficient\nCQ::Float64: torque coefficient\n\nif propeller\n\neff::Float64: efficiency\nCT::Float64: thrust coefficient\nCQ::Float64: torque coefficient\n\n\n\n\n\n"
},

{
    "location": "tutorial/#Wind-Turbine-1",
    "page": "Guide",
    "title": "Wind Turbine",
    "category": "section",
    "text": "First, we need to define parameters that are common to the rotor.  Rhub/Rtip are used for hub/tip corrections, for integration of loads, and for nondimensionalization.    The parameter turbine just changes the intput/output conventions.  If turbine=true then the following positive directions for inputs, induced velocities, and loads are used as shown in the figure below.(Image: )The definition for the precone is shown below:(Image: )RotorThis example corresponds to the NREL 5MW reference wind turbine.# --- rotor definition ---\nRhub = 1.5\nRtip = 63.0\nB = 3  # number of blades\nturbine = true\nprecone = 2.5*pi/180\n\nrotor = Rotor(Rhub, Rtip, B, turbine, precone)\n\nnothing # hideNext, we need to define the section properties.  The positive definition for twist is shown in the figure above.SectionLet\'s define all of these variables except the airfoils, which take a little more explanation.  Note that r is the distance along the blade, rather than in the rotor plane.  warning: Warning\nYou must have Rhub < r < Rtip for all r.  These are strict inequalities (not equal to).\nr = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,\n    28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,\n    56.1667, 58.9000, 61.6333]\nchord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,\n    3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]\ntheta = pi/180*[13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,\n    6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]\n\nnothing # hideThe airfoil input is any function of the form: cl, cd = airfoil(alpha, Re, Mach) where cl and cd are the lift and drag coefficient respectively, alpha is the angle of attack in radians, Re is the Reynolds number, and Mach is the Mach number.  Some of the inputs can be ignored if desired (e.g., Mach number).  While users can supply any function, two convenience methods exist for creating these functions.  One takes data from a file, the other uses input arrays.  Both convert the data into a smooth cubic spline, with some smoothing to prevent spurious jumps, as a function of angle of attack (Re and Mach currently ignored).  The required file format contains one header line that is ignored in parsing.  Its purpose is to allow you to record any information you want about the data in that file (e.g., where it came from, type of corrections applied, Reynolds number, etc.).  The rest of the file contains data in columns split by  whitespace (not commas) in the following order: alpha, cl, cd.  You can add additional columns of data (e.g., cm), but they will be ignored in this module.  Currently, only one Reynolds number is allowed.For example, a simple file (a cylinder section) would look like:Cylinder section with a Cd of 0.50.  Re = 1 million.\n-180.0   0.000   0.5000   0.000\n0.00     0.000   0.5000   0.000\n180.0    0.000   0.5000   0.000The function call is given by:af_from_file(filename)Alternatively, if you have the alpha, cl, cd data already in vectors you can initialize directly from the vectors:af_from_data(alpha, cl, cd)In this example, we will initialize from files since the data arrays would be rather long.  The only complication is that there are 8 different airfoils used at the 17 different radial stations so we need to assign them to the correct stations corresponding to the vector r defined previously.# Define airfoils.  In this case we have 8 different airfoils that we load into an array.\n# These airfoils are defined in files.\naftypes = Array{Any}(undef, 8)\naftypes[1] = af_from_file(\"airfoils/Cylinder1.dat\")\naftypes[2] = af_from_file(\"airfoils/Cylinder2.dat\")\naftypes[3] = af_from_file(\"airfoils/DU40_A17.dat\")\naftypes[4] = af_from_file(\"airfoils/DU35_A17.dat\")\naftypes[5] = af_from_file(\"airfoils/DU30_A17.dat\")\naftypes[6] = af_from_file(\"airfoils/DU25_A17.dat\")\naftypes[7] = af_from_file(\"airfoils/DU21_A17.dat\")\naftypes[8] = af_from_file(\"airfoils/NACA64_A17.dat\")\n\n# indices correspond to which airfoil is used at which station\naf_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]\n\nnothing # hideWe can now define the rotor geometry.  We will use the broadcasting syntax to create all the sections at once.sections = Section.(r, chord, theta, aftypes[af_idx])\nnothing # hideNext, we need to specify the inflow conditions.  At a basic level the inflow conditions need to be defined as a struct defined by Inflow.  The parameters mu and asound are optional if Reynolds number and Mach number respectively are used in the airfoil functions.InflowThe coordinate system for Vx and Vy is shown at the top of Wind Turbine.  In general, different inflow conditions will exist at every location along the blade. The above type allows one to specify an arbitrary input definition, however, convenience methods exist for a few typical inflow conditions.  For a typical wind turbine inflow you can use the windturbineinflow function, which is based on the angles and equations shown below.(Image: )To account for the velocity change across the hub face we compute the height of each blade location relative to the hub using coordinate transformations (where Phi is the precone angle):  z_h = r cosPhi cospsi cosTheta + r sinPhisinThetathen apply the shear exponent (alpha):  V_shear = V_hub left(1 + fracz_hH_hub right)^alphawhere H_hub is the hub height.  Finally, we can compute the x- and y-components of velocity with additional coordinate transformations:beginaligned\nV_x = V_shear ((cos gamma sin Theta cos psi + sin gamma sin psi)sin Phi + cos gamma cos Theta cos Phi)\nV_y = V_shear (cos gamma sin Thetasin psi - sin gamma cos psi) + Omega r cosPhi\nendalignedwindturbineinflowWe will use this function for this example, at a tip-speed ratio of 7.55.  Again we use a broadcasting syntax to initialize inflow conditions across the entire blade.yaw = 0.0*pi/180\ntilt = 5.0*pi/180\nhubHt = 90.0\nshearExp = 0.2\n\n# operating point for the turbine/propeller\nVinf = 10.0\ntsr = 7.55\nrotorR = Rtip*cos(precone)\nOmega = Vinf*tsr/rotorR\nazimuth = 0.0*pi/180\nrho = 1.225\n\ninflows = windturbineinflow.(Vinf, Omega, r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)\n\nnothing # hideWe have now defined the requisite inputs and can start using the BEM methodology.  The solve function is the core of the BEM.solveThe output result is a struct defined below. The positive directions for the normal and tangential forces, and the induced velocities are shown at the top of Wind Turbine.OutputsNote that we use broadcasting to solve all sections in one call.  outputs = solve.(rotor, sections, inflows)\n\nnothing # hideThere are various convenience methods to post-process the output data.  A common one is to extract the loads as arrays using the loads function:loadsNp, Tp = loads(outputs)\n\n# plot distributed loads\nfigure()\nplot(r/Rtip, Np/1e3)\nplot(r/Rtip, Tp/1e3)\nxlabel(\"r/Rtip\")\nylabel(\"distributed loads (kN/m)\")\nlegend([\"flapwise\", \"lead-lag\"])\nsavefig(\"loads-turbine.svg\"); nothing # hide(Image: )We are likely also interested in integrated loads, like thrust and torque, which are provided by the function thrusttorque.thrusttorqueT, Q = thrusttorque(rotor, sections, outputs)As used in the above example, this would give the thrust and torque assuming the inflow conditions were constant with azimuth (overly optimistic with this case at azimuth=0).  If one wanted to compute thrust and torque using azimuthal averaging you can just compute multiple inflow conditions with different azimuth angles and then average the resulting forces as shown below.azangles = pi/180*[0.0, 90.0, 180.0, 270.0]\nazinflows = windturbineinflow_az(Vinf, Omega, r, precone, yaw, tilt, azangles, hubHt, shearExp, rho)\nT, Q = thrusttorque_azavg(rotor, sections, azinflows)One final convenience function is to nondimensionalize the outputs.  The nondimensionalization uses different conventions depending on the application.  For a wind turbine the nondimensionalization is:beginaligned\nC_T = fracTq A\nC_Q = fracQq R_disk A\nC_P = fracPq A V_hub\nendalignedwherebeginaligned\nR_disk = R_tip cos(textprecone)\nA = pi R_disk^2\nq = frac12rho V_hub^2\nendalignednondimCP, CT, CQ = nondim(T, Q, Vinf, Omega, rho, rotor)\nAs a final example, let\'s create a nondimensional power curve for this turbine (power coefficient vs tip-speed-ratio):ntsr = 20  # number of tip-speed ratios\ntsrvec = range(2, 15, length=ntsr)\ncpvec = zeros(ntsr)  # initialize arrays\nctvec = zeros(ntsr)\n\nazangles = pi/180*[0.0, 90.0, 180.0, 270.0]\n\nfor i = 1:ntsr\n    Omega = Vinf*tsrvec[i]/rotorR\n\n    azinflows = windturbineinflow_az(Vinf, Omega, r, precone, yaw, tilt, azangles, hubHt, shearExp, rho)\n    T, Q = thrusttorque_azavg(rotor, sections, azinflows)\n\n    cpvec[i], ctvec[i], _ = nondim(T, Q, Vinf, Omega, rho, rotor)\nend\n\nfigure()\nplot(tsrvec, cpvec)\nplot(tsrvec, ctvec)\nxlabel(\"tip speed ratio\")\nlegend([L\"C_P\", L\"C_T\"])\nsavefig(\"cpct-turbine.svg\"); nothing # hide(Image: )"
},

{
    "location": "tutorial/#CCBlade.simpleinflow",
    "page": "Guide",
    "title": "CCBlade.simpleinflow",
    "category": "function",
    "text": "simpleinflow(Vinf, Omega, r, rho, mu=1.0, asound=1.0, precone=0.0)\n\nUniform inflow through rotor.  Returns an Inflow object.\n\nArguments\n\nVinf::Float64: freestream speed (m/s)\nOmega::Float64: rotation speed (rad/s)\nr::Float64: radial location where inflow is computed (m)\nprecone::Float64: precone angle (rad)\nrho::Float64: air density (kg/m^3)\n\n\n\n\n\n"
},

{
    "location": "tutorial/#CCBlade.effectivewake",
    "page": "Guide",
    "title": "CCBlade.effectivewake",
    "category": "function",
    "text": "effectivewake(outputs)\n\nComputes rotor wake velocities.\n\nNote that the BEM methodology applies hub/tip losses to the loads rather than to the velocities.   This is the most common way to implement a BEM, but it means that the raw velocities are misleading  as they do not contain any hub/tip loss corrections. To fix this we compute the effective hub/tip losses that would produce the same thrust/torque. In other words:\n\nCT = 4 a (1 - a) F = 4 a G (1 - a G)\n\nThis is solved for G, then multiplied against the wake velocities.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#Propellers-1",
    "page": "Guide",
    "title": "Propellers",
    "category": "section",
    "text": "The propeller analysis follows a very similar format.  The areas that are in common will not be repeated, only differences will be highlighted.using CCBlade\nusing PyPlotAgain, we first define the geometry, including the airfoils (which are the same along the blade in this case).  The positive conventions for a propeller (turbine=false) are shown in the figure below.  The underlying theory is unified across the two methods, but the input/output conventions differ to match common usage in the respective domains.(Image: )\n# rotor definition\nRhub = 0.0254*.5\nRtip = 0.0254*3.0\nB = 2  # number of blades\nturbine = false\n\nrotor = Rotor(Rhub, Rtip, B, turbine)\n\n# section definition\n\nr = .0254*[0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332,\n     1.1128, 1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911,\n     1.6708, 1.7505, 1.8302, 1.9099, 1.9896, 2.0693, 2.1490, 2.2287,\n     2.3084, 2.3881, 2.4678, 2.5475, 2.6273, 2.7070, 2.7867, 2.8661, 2.9410]\nchord = .0254*[0.6270, 0.6255, 0.6231, 0.6199, 0.6165, 0.6125, 0.6054, 0.5973, 0.5887,\n          0.5794, 0.5695, 0.5590, 0.5479, 0.5362, 0.5240, 0.5111, 0.4977,\n          0.4836, 0.4689, 0.4537, 0.4379, 0.4214, 0.4044, 0.3867, 0.3685,\n          0.3497, 0.3303, 0.3103, 0.2897, 0.2618, 0.1920]\ntheta = pi/180.0*[40.2273, 38.7657, 37.3913, 36.0981, 34.8803, 33.5899, 31.6400,\n                   29.7730, 28.0952, 26.5833, 25.2155, 23.9736, 22.8421, 21.8075,\n                   20.8586, 19.9855, 19.1800, 18.4347, 17.7434, 17.1005, 16.5013,\n                   15.9417, 15.4179, 14.9266, 14.4650, 14.0306, 13.6210, 13.2343,\n                   12.8685, 12.5233, 12.2138]\n\naf = af_from_file(\"airfoils/NACA64_A17.dat\")\n\nsections = Section.(r, chord, theta, af)\nnothing # hideNext, we define the inflow.  For a propeller, it typically doesn\'t operate with tilt, yaw, and shear like a wind turbine does, so we have defined another convenience function for simple uniform inflow.  Like before, you can always define your own arbitrary inflow object.simpleinflowIn this example, we assume simple inflow.  Again, note the broadcast syntax to get inflows at all sections.\nrho = 1.225\n\nVinf = 10.0\nOmega = 8000.0*pi/30.0\n\ninflows = simpleinflow.(Vinf, Omega, r, rho)\nnothing # hideWe can now computed distributed loads and induced velocities.  outputs = solve.(rotor, sections, inflows)\nNp, Tp = loads(outputs)\n\nfigure()\nplot(r/Rtip, Np)\nplot(r/Rtip, Tp)\nxlabel(\"r/Rtip\")\nylabel(\"distributed loads (N/m)\")\nlegend([\"flapwise\", \"lead-lag\"])\nsavefig(\"loads-prop.svg\"); nothing # hide(Image: )This time we will also look at the induced velocities.  This is usually not of interest for wind turbines, but for propellers can be useful to assess, for example, prop-on-wing interactions.effectivewakeu, v = effectivewake(outputs)\n\nfigure()\nplot(r/Rtip, u/Vinf)\nplot(r/Rtip, v/Vinf)\nxlabel(\"r/Rtip\")\nylabel(\"(normalized) induced velocity at rotor disk\")\nlegend([\"axial velocity\", \"swirl velocity\"])\nsavefig(\"velocity-prop.svg\"); nothing # hide(Image: )As before, we\'d like to evaluate integrated quantities at multiple conditions in a for loop (advance ratios as is convention for propellers instead of tip-speed ratios).  The normalization conventions for a propeller are:beginaligned\nC_T = fracTrho n^2 D^4\nC_Q = fracQrho n^2 D^5\nC_P = fracPrho n^3 D^5 = fracC_Q2 pi\neta = fracC_T JC_P\nendalignedwherebeginaligned\nn = fracOmega2pi text rev per sec\nD = 2 R_tip cos(textprecone)\nJ = fracVn D\nendalignednote: Note\nEfficiency is set to zero if the thrust is negative (producing drag).  The code below performs this analysis then plots thrust coefficient, power coefficient, and efficiency as a function of advance ratio.nJ = 20  # number of advance ratios\n\nJ = range(0.1, 0.9, length=nJ)  # advance ratio\n\nOmega = 8000.0*pi/30\nn = Omega/(2*pi)\nD = 2*Rtip\n\neff = zeros(nJ)\nCT = zeros(nJ)\nCQ = zeros(nJ)\n\nfor i = 1:nJ\n    Vinf = J[i] * D * n\n\n    inflows = simpleinflow.(Vinf, Omega, r, rho)\n    outputs = solve.(rotor, sections, inflows)\n    T, Q = thrusttorque(rotor, sections, outputs)\n    eff[i], CT[i], CQ[i] = nondim(T, Q, Vinf, Omega, rho, rotor)\n\nend\n\nfigure()\nplot(J, CT)\nplot(J, CQ*2*pi)\nxlabel(L\"J\")\nlegend([L\"C_T\", L\"C_P\"])\nsavefig(\"ctcp-prop.svg\") # hide\n\nfigure()\nplot(J, eff)\nxlabel(L\"J\")\nylabel(L\"\\eta\")\nsavefig(\"eta-prop.svg\"); nothing # hide(Image: ) (Image: )"
},

]}
