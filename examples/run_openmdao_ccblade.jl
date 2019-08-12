using OpenMDAO
using PyCall
using CCBlade
using LinearAlgebra: norm
using PyPlot

julia_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")
pyccblade = pyimport("ccblade.ccblade")
pyccblade_geom = pyimport("ccblade.geometry")
get_rows_cols = pyccblade.get_rows_cols

# Need to create a CCBlade component.
function ccblade_residual_apply_nonlinear!(options, inputs, outputs, residuals)
    # Airfoil interpolation object.
    af = options["af"]

    # Rotor parameters.
    B = options["B"]
    Rhub = inputs["hub_radius"][1]
    Rtip = inputs["prop_radius"][1]
    precone = inputs["precone"][1]
    turbine = options["turbine"]
    rotor = Rotor(Rhub, Rtip, B, turbine)

    # Blade section parameters.
    r = inputs["radii"]
    chord = inputs["chord"]
    theta = inputs["theta"]
    sections = Section.(r, chord, theta, af)

    # Inflow parameters.
    Vx = inputs["Vx"]
    Vy = inputs["Vy"]
    rho = inputs["rho"]
    mu = inputs["mu"]
    asound = inputs["asound"]
    inflows = Inflow.(Vx, Vy, rho, mu, asound)

    phis = outputs["phi"]

    # Get the residual, and the outputs. The `out` variable is
    # two-dimensional array of length-two tuples. The first tuple
    # entry is the residual, and the second is the `Outputs`
    # struct.
    out = CCBlade.residual.(phis, sections, inflows, rotor)

    # Store the residual.
    @. residuals["phi"] = getindex(out, 1)

    # Get the other outputs.
    @. residuals["Np"] = outputs["Np"] - getfield(getindex(out, 2), :Np)
    @. residuals["Tp"] = outputs["Tp"] - getfield(getindex(out, 2), :Tp)
    @. residuals["a"] = outputs["a"] - getfield(getindex(out, 2), :a)
    @. residuals["ap"] = outputs["ap"] - getfield(getindex(out, 2), :ap)
    @. residuals["u"] = outputs["u"] - getfield(getindex(out, 2), :u)
    @. residuals["v"] = outputs["v"] - getfield(getindex(out, 2), :v)
    @. residuals["W"] = outputs["W"] - getfield(getindex(out, 2), :W)
    @. residuals["cl"] = outputs["cl"] - getfield(getindex(out, 2), :cl)
    @. residuals["cd"] = outputs["cd"] - getfield(getindex(out, 2), :cd)
    @. residuals["F"] = outputs["F"] - getfield(getindex(out, 2), :F)
end

ccblade_residual_apply_nonlinear2! = pyfunction(ccblade_residual_apply_nonlinear!,
                                                PyDict{String, PyAny},
                                                PyDict{String, PyArray},
                                                PyDict{String, PyArray},
                                                PyDict{String, PyArray})

function ccblade_residual_linearize!(options, inputs, outputs, partials)
    num_nodes = options["num_nodes"]
    num_radial = options["num_radial"]
    # Airfoil interpolation object.
    af = options["af"]

    # Rotor parameters.
    Rhub = inputs["hub_radius"][1]
    Rtip = inputs["prop_radius"][1]
    precone = inputs["precone"][1]
    B = options["B"]
    turbine = options["turbine"]
    rotor = Rotor(Rhub, Rtip, B, turbine)

    # Blade section parameters.
    r = inputs["radii"]
    chord = inputs["chord"]
    theta = inputs["theta"]
    sections = Section.(r, chord, theta, af)

    # Inflow parameters.
    Vx = inputs["Vx"]
    Vy = inputs["Vy"]
    rho = inputs["rho"]
    mu = inputs["mu"]
    asound = inputs["asound"]
    inflows = Inflow.(Vx, Vy, rho, mu, asound)

    # Phi, the implicit variable.
    phis = outputs["phi"]

    # Get the derivatives of the residual.
    residual_derivs = CCBlade.residual_partials.(phis, sections, inflows, rotor)

    # Copy the derivatives of the residual to the partials dict.
    wrt_name_sym = [
        ("phi", :phi),
        ("radii", :r),
        ("chord", :chord), 
        ("theta", :theta), 
        ("Vx", :Vx),
        ("Vy", :Vy),
        ("rho", :rho),
        ("mu", :mu),
        ("asound", :asound),
        ("hub_radius", :Rhub),
        ("prop_radius", :Rtip),
        ("precone", :precone)]
    for (name, sym) in wrt_name_sym
        # reshape does not copy data: https://github.com/JuliaLang/julia/issues/112
        # deriv = reshape(partials["phi", name], (num_nodes, num_radial))
        deriv = transpose(reshape(partials["phi", name], (num_radial, num_nodes)))
        @. deriv = getfield(residual_derivs, sym)
    end

    # Get the derivatives of the outputs.
    output_derivs = CCBlade.output_partials.(phis, sections, inflows, rotor)
    # @show getindex.(output_derivs, 1, 1)

    # Holy Guido, so ugly...
    of_names = String["Np", "Tp", "a", "ap", "u", "v", "phi", "W",
                      "cl", "cd", "F"]
    wrt_names = String["phi", "radii", "chord", "theta", "Vx",
                       "Vy", "rho", "mu", "asound", "hub_radius",
                       "prop_radius", "precone"]
    for (of_idx, of_name) in enumerate(of_names)
        if of_name == "phi"
            continue
        else
            for (wrt_idx, wrt_name) in enumerate(wrt_names)
                # reshape does not copy data: https://github.com/JuliaLang/julia/issues/112
                # deriv = reshape(partials[of_name, wrt_name], (num_nodes, num_radial))
                deriv = transpose(reshape(partials[of_name, wrt_name], (num_radial, num_nodes)))
                @. deriv = -getindex(output_derivs, of_idx, wrt_idx)
                # if of_name == "Np" && wrt_name == "phi"
                #     @show of_name, wrt_name, partials[of_name, wrt_name]
                # end
            end
        end
    end

end

ccblade_residual_linearize2! = pyfunction(
    ccblade_residual_linearize!,
    PyDict{String, PyAny}, # options
    PyDict{String, PyArray}, # inputs
    PyDict{String, PyArray}, # outputs
    PyDict{Tuple{String, String}, PyArray}) # partials

function ccblade_residual_guess_nonlinear!(options, inputs, outputs, residuals)
    DEBUG_PRINT = options["debug_print"]

    # These are all the explicit output names.
    out_names = ["Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "F"]

    # If the residual norm is small, we're close enough, so return.
    GUESS_TOL = 1e-4
    ccblade_residual_apply_nonlinear!(options, inputs, outputs, residuals)
    res_norm = norm(residuals["phi"])
    if res_norm < GUESS_TOL
        for name in out_names
            if all(isfinite.(residuals[name]))
                @. outputs[name] = outputs[name] - residuals[name]
            end
        end
        if DEBUG_PRINT
            println(
                "guess_nonlinear res_norm: $(res_norm) (skipping guess_nonlinear bracketing)")
        end
        return
    end

    # Airfoil interpolation object.
    af = options["af"]

    # Rotor parameters.
    B = options["B"]
    Rhub = inputs["hub_radius"][1]
    Rtip = inputs["prop_radius"][1]
    precone = inputs["precone"][1]
    turbine = options["turbine"]
    rotor = Rotor(Rhub, Rtip, B, turbine)

    # Blade section parameters.
    r = inputs["radii"]
    chord = inputs["chord"]
    theta = inputs["theta"]
    sections = Section.(r, chord, theta, af)

    # Inflow parameters.
    Vx = inputs["Vx"]
    Vy = inputs["Vy"]
    rho = inputs["rho"]
    mu = inputs["mu"]
    asound = inputs["asound"]
    inflows = Inflow.(Vx, Vy, rho, mu, asound)

    # Find an interval for each section that brackets the root
    # (hopefully). This will return a 2D array of Bool, Float64,
    # Float64 tuples.
    out = CCBlade.firstbracket.(rotor, sections, inflows)

    # Check that we bracketed the root for each section.
    success = getindex.(out, 1)
    if ! all(success)
        println("CCBlade bracketing failed")
        @warn "CCBlade bracketing failed"
    end

    # Get the left and right value of each interval.
    phi_1 = getindex.(out, 2)
    phi_2 = getindex.(out, 3)

    # Initialize the residual arrays.
    res_1 = similar(phi_1)
    res_2 = similar(phi_1)

    # Evaluate the residual for phi_1.
    @. outputs["phi"] = phi_1
    ccblade_residual_apply_nonlinear!(options, inputs, outputs, residuals)
    @. res_1 = residuals["phi"]

    # Evaluate the residual for phi_2.
    @. outputs["phi"] = phi_2
    ccblade_residual_apply_nonlinear!(options, inputs, outputs, residuals)
    @. res_2 = residuals["phi"]

    # Sort the phi_1, phi_2 values by whether they give a negative or positive
    # residual.
    mask = res_1 .> 0.0
    phi_1[mask], phi_2[mask] = phi_2[mask], phi_1[mask]

    # Sort the res_1 and res_2 by whether they are negative or positive.
    res_1[mask], res_2[mask] = res_2[mask], res_1[mask]

    if DEBUG_PRINT
        println("0 Still bracking a root? $(all((res_1.*res_2) .< 0.0))")
    end

    for i in 1:100
        # Evaulate the residual at a new phi value.
        @. outputs["phi"] = 0.5 * (phi_1 + phi_2)
        ccblade_residual_apply_nonlinear!(options, inputs, outputs, residuals)
        new_res = residuals["phi"]

        # Check if the residual satisfies the tolerance.
        res_norm = norm(new_res)
        if res_norm < GUESS_TOL
            for name in out_names
                if all(isfinite.(residuals[name]))
                    @. outputs[name] = outputs[name] - residuals[name]
                end
            end
            if DEBUG_PRINT
                println(
                    "guess_nonlinear res_norm: $(res_norm), convergence criteria satisfied")
            end
            return
        end

        # Sort the phi and residual values again.
        mask_1 = new_res .< 0
        mask_2 = new_res .> 0

        phi_1[mask_1] = outputs["phi"][mask_1]
        res_1[mask_1] = new_res[mask_1]

        phi_2[mask_2] = outputs["phi"][mask_2]
        res_2[mask_2] = new_res[mask_2]

        if DEBUG_PRINT
            println("$(i+1) res_norm = $(res_norm), Still bracking a root? $(all((res_1.*res_2) .< 0.0))")
        end
    end

    # If we get here, we were unable to satisfy the convergence criteria.
    for name in out_names
        if all(isfinite.(residuals[name]))
            # outputs[name] -= residuals[name]
            @. outputs[name] = outputs[name] - residuals[name]
        end
    end
    if DEBUG_PRINT
        println(
            "guess_nonlinear res_norm: $(res_norm) > GUESS_TOL")
    end

end

ccblade_residual_guess_nonlinear2! = pyfunction(
    ccblade_residual_guess_nonlinear!,
    PyDict{String, PyAny}, # options
    PyDict{String, PyArray}, # inputs
    PyDict{String, PyArray}, # outputs
    PyDict{String, PyArray}) # residuals)

num_nodes = 1
num_blades = 3
num_radial = 15

af_filename = "airfoils/mh117.dat"
af = af_from_file(af_filename, use_interpolations_jl=true)

ccblade_residual_options_data = [
    OptionsData("num_nodes", Int, num_nodes),
    OptionsData("num_radial", Int, num_radial),
    OptionsData("B", Int, num_blades),
    OptionsData("af", nothing, af),
    OptionsData("turbine", Bool, false),
    OptionsData("debug_print", Bool, true)]

ccblade_residual_input_data = [
    VarData("radii", [1, num_radial], 1., "m")
    VarData("chord", [1, num_radial], 1., "m")
    VarData("theta", [1, num_radial], 1., "rad")
    VarData("Vx", [num_nodes, 1], 1., "m/s")
    VarData("Vy", [num_nodes, num_radial], 1., "m/s")
    VarData("rho", [num_nodes, 1], 1., "kg/m**3")
    VarData("mu", [num_nodes, 1], 1., "N/m**2*s")
    VarData("asound", [num_nodes, 1], 1., "m/s")
    VarData("hub_radius", [1, 1], 1., "m")
    VarData("prop_radius", [1, 1], 1., "m")
    VarData("precone", [1, 1], 1., "rad")]

ccblade_residual_output_data = [
    VarData("phi", [num_nodes, num_radial], 1., "rad"),
    VarData("Np", [num_nodes, num_radial], 1., "N/m"),
    VarData("Tp", [num_nodes, num_radial], 1., "N/m"),
    VarData("a", [num_nodes, num_radial], 1.),
    VarData("ap", [num_nodes, num_radial], 1.),
    VarData("u", [num_nodes, num_radial], 1., "m/s"),
    VarData("v", [num_nodes, num_radial], 1., "m/s"),
    VarData("W", [num_nodes, num_radial], 1., "m/s"),
    VarData("cl", [num_nodes, num_radial], 1.),
    VarData("cd", [num_nodes, num_radial], 1.),
    VarData("F", [num_nodes, num_radial], 1.)]

ccblade_residual_partials_data = Array{PartialsData, 1}()

rows, cols = get_rows_cols(
    of_shape=(num_nodes, num_radial), of_ss="ij",
    wrt_shape=(1,), wrt_ss="k")
of_names = ["phi", "Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "F"]
for name in of_names
    push!(ccblade_residual_partials_data, PartialsData(name, "hub_radius", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "prop_radius", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "precone", rows=rows, cols=cols))
end

rows, cols = get_rows_cols(
    of_shape=(num_nodes, num_radial), of_ss="ij",
    wrt_shape=(num_radial,), wrt_ss="j")
for name in of_names
    push!(ccblade_residual_partials_data, PartialsData(name, "radii", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "chord", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "theta", rows=rows, cols=cols))
end

rows, cols = get_rows_cols(
    of_shape=(num_nodes, num_radial), of_ss="ij",
    wrt_shape=(num_nodes,), wrt_ss="i")
for name in of_names
    push!(ccblade_residual_partials_data, PartialsData(name, "Vx", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "rho", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "mu", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "asound", rows=rows, cols=cols))
end

rows, cols = get_rows_cols(
    of_shape=(num_nodes, num_radial), of_ss="ij",
    wrt_shape=(num_nodes, num_radial), wrt_ss="ij")
for name in of_names
    push!(ccblade_residual_partials_data, PartialsData(name, "Vy", rows=rows, cols=cols))
    push!(ccblade_residual_partials_data, PartialsData(name, "phi", rows=rows, cols=cols))
end

for name in ["Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "F"]
    push!(ccblade_residual_partials_data, PartialsData(name, name, rows=rows, cols=cols, val=1.0))
end

ccblade_residual_comp_data = ICompData(
    ccblade_residual_input_data,
    ccblade_residual_output_data,
    options=ccblade_residual_options_data,
    partials=ccblade_residual_partials_data,
    apply_nonlinear=ccblade_residual_apply_nonlinear2!,
    linearize=ccblade_residual_linearize2!,
    guess_nonlinear=ccblade_residual_guess_nonlinear2!)

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
                   v=Dict("units" => "m/s", "shape" => num_nodes),
                   precone=Dict("units" => "rad"),
                   Vx=Dict("units" => "m/s", "shape" => num_nodes))
group.add_subsystem("Vx_comp", comp, promotes=["*"])

comp = om.ExecComp("Vy = omega*radii*cos(precone)",
                   omega=Dict("units" => "rad/s",
                          "shape" => (num_nodes, 1),
                          "flat_src_indices" => [0]),
                   radii=Dict("units" => "m", "shape" => (1, num_radial)),
                   precone=Dict("units" => "rad"),
                   Vy=Dict("units" => "m/s", "shape" => (num_nodes, num_radial)))
group.add_subsystem("Vy_comp", comp, promotes=["*"])

comp = julia_comps.JuliaImplicitComp(julia_comp_data=ccblade_residual_comp_data)
comp.linear_solver = om.DirectSolver(assemble_jac=true)
comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=true, iprint=2, err_on_non_converge=true)
group.add_subsystem("ccblade_residual_comp", comp,
                    promotes_inputs=["radii", "chord", "theta", "Vx", "Vy",
                                     "rho", "mu", "asound", "hub_radius",
                                     "prop_radius", "precone"],
                    promotes_outputs=["Np", "Tp"])

comp = pyccblade.CCBladeThrustTorqueComp(num_nodes=num_nodes,
                                         num_radial=num_radial, B=num_blades)
group.add_subsystem("ccblade_torquethrust_comp", comp,
                    promotes_inputs=["radii", "dradii", "Np", "Tp"],
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
                     "prop_diameter"],
    promotes_outputs=["thrust", "torque", "efficiency"])

prob.model.add_design_var("chord_dv", lower=1., upper=20.,
                          scaler=5e-2)
prob.model.add_design_var("theta_dv",
                          lower=20.0*pi/180., upper=90.0*pi/180.0)

prob.model.add_objective("efficiency", scaler=-1.,)
prob.model.add_constraint("thrust", equals=700.0, scaler=1e-3,
                          indices=collect(0:num_nodes-1))
# This doesn't work. Not sure why. Looks like it doesn't get copied to the
# Python side of things, so OpenMDAO uses the default driver (SLSQP).
prob.driver = om.pyOptSparseDriver()
# prob.driver.options["optimizer"] = "SNOPT"
# This does work.
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
