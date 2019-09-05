import Base.convert
using PyCall
using OpenMDAO
using LinearAlgebra: norm

struct CCBladeResidualComp
    num_nodes
    num_radial
    af
    B
    turbine
    debug_print
    inputs
    outputs
    partials
end

convert(::Type{CCBladeResidualComp}, po::PyObject) = CCBladeResidualComp(
    po.num_nodes, po.num_radial, po.af, po.B, po.turbine, po.debug_print,
    po.inputs, po.outputs, po.partials)

function CCBladeResidualComp(; num_nodes, num_radial, af, B, turbine, debug_print)

    # Putting these two lines at the top level of this file doesn't work:
    # get_rows_cols will be a "Null" PyObject, and then the first call to
    # get_rows_cols will give a nasty segfault.
    pyccblade = pyimport("ccblade.ccblade_jl")
    get_rows_cols = pyccblade.get_rows_cols

    input_data = [
        VarData("r", [1, num_radial], 1., "m")
        VarData("chord", [1, num_radial], 1., "m")
        VarData("theta", [1, num_radial], 1., "rad")
        VarData("Vx", [num_nodes, num_radial], 1., "m/s")
        VarData("Vy", [num_nodes, num_radial], 1., "m/s")
        VarData("rho", [num_nodes, 1], 1., "kg/m**3")
        VarData("mu", [num_nodes, 1], 1., "N/m**2*s")
        VarData("asound", [num_nodes, 1], 1., "m/s")
        VarData("Rhub", [1, 1], 1., "m")
        VarData("Rtip", [1, 1], 1., "m")
        VarData("precone", [1, 1], 1., "rad")]

    output_data = [
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

    partials_data = Array{PartialsData, 1}()

    rows, cols = get_rows_cols(
        of_shape=(num_nodes, num_radial), of_ss="ij",
        wrt_shape=(1,), wrt_ss="k")
    of_names = ["phi", "Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "F"]
    for name in of_names
        push!(partials_data, PartialsData(name, "Rhub", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "Rtip", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "precone", rows=rows, cols=cols))
    end

    rows, cols = get_rows_cols(
        of_shape=(num_nodes, num_radial), of_ss="ij",
        wrt_shape=(num_radial,), wrt_ss="j")
    for name in of_names
        push!(partials_data, PartialsData(name, "r", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "chord", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "theta", rows=rows, cols=cols))
    end

    rows, cols = get_rows_cols(
        of_shape=(num_nodes, num_radial), of_ss="ij",
        wrt_shape=(num_nodes,), wrt_ss="i")
    for name in of_names
        push!(partials_data, PartialsData(name, "rho", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "mu", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "asound", rows=rows, cols=cols))
    end

    rows, cols = get_rows_cols(
        of_shape=(num_nodes, num_radial), of_ss="ij",
        wrt_shape=(num_nodes, num_radial), wrt_ss="ij")
    for name in of_names
        push!(partials_data, PartialsData(name, "Vx", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "Vy", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "phi", rows=rows, cols=cols))
    end

    for name in ["Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "F"]
        push!(partials_data, PartialsData(name, name, rows=rows, cols=cols, val=1.0))
    end

    return CCBladeResidualComp(num_nodes, num_radial, af, B, turbine, debug_print, input_data, output_data, partials_data)
end

function OpenMDAO.apply_nonlinear!(self::CCBladeResidualComp, inputs, outputs, residuals)
    # Airfoil interpolation object.
    af = self.af

    # Rotor parameters.
    B = self.B
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    precone = inputs["precone"][1]
    turbine = self.turbine
    rotor = Rotor(Rhub, Rtip, B, turbine)

    # Blade section parameters.
    r = inputs["r"]
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

function OpenMDAO.linearize!(self::CCBladeResidualComp, inputs, outputs, partials)
    num_nodes = self.num_nodes
    num_radial = self.num_radial
    # Airfoil interpolation object.
    af = self.af

    # Rotor parameters.
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    precone = inputs["precone"][1]
    B = self.B
    turbine = self.turbine
    rotor = Rotor(Rhub, Rtip, B, turbine)

    # Blade section parameters.
    r = inputs["r"]
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
        ("r", :r),
        ("chord", :chord), 
        ("theta", :theta), 
        ("Vx", :Vx),
        ("Vy", :Vy),
        ("rho", :rho),
        ("mu", :mu),
        ("asound", :asound),
        ("Rhub", :Rhub),
        ("Rtip", :Rtip),
        ("precone", :precone)]
    for (name, sym) in wrt_name_sym
        # reshape does not copy data: https://github.com/JuliaLang/julia/issues/112
        deriv = transpose(reshape(partials["phi", name], (num_radial, num_nodes)))
        @. deriv = getfield(residual_derivs, sym)
    end

    # Get the derivatives of the outputs.
    output_derivs = CCBlade.output_partials.(phis, sections, inflows, rotor)
    # @show getindex.(output_derivs, 1, 1)

    # Holy Guido, so ugly...
    of_names = String["Np", "Tp", "a", "ap", "u", "v", "phi", "W",
                      "cl", "cd", "F"]
    wrt_names = String["phi", "r", "chord", "theta", "Vx",
                       "Vy", "rho", "mu", "asound", "Rhub",
                       "Rtip", "precone"]
    for (of_idx, of_name) in enumerate(of_names)
        if of_name == "phi"
            continue
        else
            for (wrt_idx, wrt_name) in enumerate(wrt_names)
                # reshape does not copy data: https://github.com/JuliaLang/julia/issues/112
                deriv = transpose(reshape(partials[of_name, wrt_name], (num_radial, num_nodes)))
                @. deriv = -getindex(output_derivs, of_idx, wrt_idx)
            end
        end
    end

end

function OpenMDAO.guess_nonlinear!(self::CCBladeResidualComp, inputs, outputs, residuals)
    DEBUG_PRINT = self.debug_print

    # These are all the explicit output names.
    out_names = ["Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "F"]

    # If the residual norm is small, we're close enough, so return.
    GUESS_TOL = 1e-4
    OpenMDAO.apply_nonlinear!(self, inputs, outputs, residuals)
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
    af = self.af

    # Rotor parameters.
    B = self.B
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    precone = inputs["precone"][1]
    turbine = self.turbine
    rotor = Rotor(Rhub, Rtip, B, turbine)

    # Blade section parameters.
    r = inputs["r"]
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
    OpenMDAO.apply_nonlinear!(self, inputs, outputs, residuals)
    @. res_1 = residuals["phi"]

    # Evaluate the residual for phi_2.
    @. outputs["phi"] = phi_2
    OpenMDAO.apply_nonlinear!(self, inputs, outputs, residuals)
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
        OpenMDAO.apply_nonlinear!(self, inputs, outputs, residuals)
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
            @. outputs[name] = outputs[name] - residuals[name]
        end
    end
    if DEBUG_PRINT
        println(
            "guess_nonlinear res_norm: $(res_norm) > GUESS_TOL")
    end

end

function OpenMDAO.solve_nonlinear!(self::CCBladeResidualComp, inputs, outputs)
    # Airfoil interpolation object.
    af = self.af

    # Rotor parameters.
    B = self.B
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    precone = inputs["precone"][1]
    turbine = self.turbine
    rotor = Rotor(Rhub, Rtip, B, turbine)

    # Blade section parameters.
    r = inputs["r"]
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

    # When called this way, CCBlade.solve will return a 2D array of `Outputs`
    # objects.
    out = CCBlade.solve.(rotor, sections, inflows)

    # Set the outputs.
    @. outputs["phi"] = getfield(out, :phi)
    @. outputs["Np"] = getfield(out, :Np)
    @. outputs["Tp"] = getfield(out, :Tp)
    @. outputs["a"] = getfield(out, :a)
    @. outputs["ap"] = getfield(out, :ap)
    @. outputs["u"] = getfield(out, :u)
    @. outputs["v"] = getfield(out, :v)
    @. outputs["W"] = getfield(out, :W)
    @. outputs["cl"] = getfield(out, :cl)
    @. outputs["cd"] = getfield(out, :cd)
    @. outputs["F"] = getfield(out, :F)

end
