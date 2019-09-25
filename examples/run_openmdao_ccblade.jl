using OpenMDAO
using PyCall
using CCBlade
using PyPlot

om = pyimport("openmdao.api")
pyccblade = pyimport("ccblade.ccblade_jl")
pyccblade_py = pyimport("ccblade.ccblade_py")
pyccblade_geom = pyimport("ccblade.geometry")

num_nodes = 1
num_blades = 3
num_radial = 15

af_filename = "airfoils/mh117.dat"
af = af_from_file(af_filename, use_interpolations_jl=true)

num_cp = 6
chord = 10.0
theta = collect(LinRange(65., 25., num_cp)) .* pi/180.0
pitch = 0.0

hub_diameter = 30.  # cm
prop_diameter = 150.  # cm
c0 = sqrt(1.4*287.058*300.0)  # meters/second
rho0 = 1.4*98600.0/(c0*c0)  # kg/m^3
omega = 236.0

prob = om.Problem()

comp = om.IndepVarComp()
comp.add_discrete_output("B", val=num_blades)
comp.add_output("rho", val=rho0, shape=num_nodes, units="kg/m**3")
comp.add_output("mu", val=1., shape=num_nodes, units="N/m**2*s")
comp.add_output("asound", val=c0, shape=num_nodes, units="m/s")
comp.add_output("v", val=77.2, shape=num_nodes, units="m/s")
comp.add_output("alpha", val=0., shape=num_nodes, units="rad")
comp.add_output("incidence", val=0., shape=num_nodes, units="rad")
comp.add_output("precone", val=0., units="deg")
comp.add_output("omega", val=omega, shape=num_nodes, units="rad/s")
comp.add_output("hub_diameter", val=hub_diameter, shape=num_nodes, units="cm")
comp.add_output("prop_diameter", val=prop_diameter, shape=num_nodes, units="cm")
comp.add_output("pitch", val=pitch, shape=num_nodes, units="rad")
comp.add_output("chord_dv", val=chord, shape=num_cp, units="cm")
comp.add_output("theta_dv", val=theta, shape=num_cp, units="rad")
prob.model.add_subsystem("indep_var_comp", comp, promotes=["*"])

comp = pyccblade_geom.GeometryGroup(num_nodes=num_nodes, num_cp=num_cp, num_radial=num_radial)
prob.model.add_subsystem(
    "geometry_group", comp,
    promotes_inputs=["hub_diameter", "prop_diameter", "chord_dv",
                     "theta_dv", "pitch"],
    promotes_outputs=["radii", "dradii", "chord", "theta"])

group = om.Group()

comp = om.ExecComp("hub_radius = 0.5*hub_diameter",
                   hub_radius=Dict("value" => 0.1, "units" => "m"),
                   hub_diameter=Dict("units" => "m"))
group.add_subsystem("hub_radius_comp", comp, promotes=["*"])

comp = om.ExecComp("prop_radius = 0.5*prop_diameter",
                   prop_radius=Dict("value" => 1.0, "units" => "m"),
                   prop_diameter=Dict("units" => "m"))
group.add_subsystem("prop_radius_comp", comp, promotes=["*"])

comp = om.ExecComp("Vx = v*cos(precone)",
                   v=Dict("units" => "m/s",
                          "shape" => (num_nodes, num_radial),
                          "src_indices" => repeat(0:num_nodes-1, num_nodes, num_radial)),
                   precone=Dict("units" => "rad"),
                   Vx=Dict("units" => "m/s", "shape" => (num_nodes, num_radial)))
group.add_subsystem("Vx_comp", comp, promotes=["*"])

comp = om.ExecComp("Vy = omega*radii*cos(precone)",
                   omega=Dict("units" => "rad/s",
                          "shape" => (num_nodes, 1),
                          "flat_src_indices" => collect(0:num_nodes-1)),
                   radii=Dict("units" => "m", "shape" => (1, num_radial)),
                   precone=Dict("units" => "rad"),
                   Vy=Dict("units" => "m/s", "shape" => (num_nodes, num_radial)))
group.add_subsystem("Vy_comp", comp, promotes=["*"])

ccblade_residual_comp_data = CCBlade.CCBladeResidualComp(
    num_nodes=num_nodes, num_radial=num_radial, af=af, B=num_blades, turbine=false, debug_print=true)
comp = make_component(ccblade_residual_comp_data)
comp.linear_solver = om.DirectSolver(assemble_jac=true)
# comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=true, iprint=2, err_on_non_converge=true)
group.add_subsystem("ccblade_residual_comp", comp,
                    promotes_inputs=[("r", "radii"), "chord", "theta", "Vx",
                                     "Vy", "rho", "mu", "asound",
                                     # "pitch",
                                     ("Rhub", "hub_radius"), ("Rtip", "prop_radius"),
                                     "precone"],
                    promotes_outputs=["Np", "Tp"])

comp = pyccblade_py.FunctionalsComp(num_nodes=num_nodes, num_radial=num_radial)
group.add_subsystem("ccblade_torquethrust_comp", comp,
                    promotes_inputs=["B", "radii", "dradii", "Np", "Tp"],
                    promotes_outputs=["thrust", "torque"])

comp = om.ExecComp("efficiency = (thrust*v)/(torque*omega)",
                   thrust=Dict("units" => "N", "shape" => num_nodes),
                   v=Dict("units" => "m/s", "shape" => num_nodes),
                   torque=Dict("units" => "N*m", "shape" => num_nodes),
                   omega=Dict("units" => "rad/s", "shape" => num_nodes),
                   efficiency=Dict("shape" => num_nodes))
group.add_subsystem("efficiency_comp", comp,
                    promotes_inputs=["*"],
                    promotes_outputs=["*"])

group.linear_solver = om.DirectSolver(assemble_jac=true)

prob.model.add_subsystem(
    "ccblade_group", group,
    promotes_inputs=["radii", "dradii", "chord", "theta", "rho", "mu",
                     "asound", "v", "precone", "omega", "hub_diameter",
                     "prop_diameter",
                     # "pitch",
                    ],
    promotes_outputs=["thrust", "torque", "efficiency"])

prob.model.add_design_var("chord_dv", lower=1., upper=20.,
                          scaler=5e-2)
prob.model.add_design_var("theta_dv",
                          lower=20.0*pi/180., upper=90.0*pi/180.0)

prob.model.add_objective("efficiency", scaler=-1.,)
prob.model.add_constraint("thrust", equals=700.0, scaler=1e-3,
                          indices=collect(0:num_nodes-1))
prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")

prob.setup()
prob.final_setup()
prob.run_driver()

node = 1
radii = prob.get_val("radii", units="m")[node, :]
ccblade_normal_load = prob.get_val(
    "ccblade_group.Np", units="N/m")[node, :]*num_blades
ccblade_circum_load = prob.get_val(
    "ccblade_group.Tp", units="N/m")[node, :]*num_blades

fig, ax = plt.subplots()
plot(radii, ccblade_normal_load, label="CCBlade.jl")
xlabel("blade element radius, m")
ylabel("normal load, N/m")
legend()
savefig("normal_load.png")

fig, ax = plt.subplots()
plot(radii, ccblade_circum_load, label="CCBlade.jl")
xlabel("blade element radius, m")
ylabel("circumferential load, N/m")
legend()
savefig("circum_load.png")
