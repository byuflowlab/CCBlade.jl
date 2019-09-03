import Base.convert
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

convert(::Type{CCBladeResidualComp}, po::PyCall.PyObject) = CCBladeResidualComp(
    po.num_nodes, po.num_radial, po.af, po.B, po.turbine, po.debug_print,
    po.inputs, po.outputs, po.partials)

function CCBladeResidualComp(; num_nodes, num_radial, af, B, turbine, debug_print)

    # Putting these two lines at the top level of this file doesn't work:
    # get_rows_cols will be a "Null" PyObject, and then the first call to
    # get_rows_cols will give a nasty segfault. I should just rewrite
    # get_rows_cols in Julia.
    pyccblade = PyCall.pyimport("ccblade.ccblade_jl")
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
        VarData("pitch", [num_nodes, 1], 1., "rad")
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
        VarData("alpha", [num_nodes, num_radial], 1., "rad"),
        VarData("W", [num_nodes, num_radial], 1., "m/s"),
        VarData("cl", [num_nodes, num_radial], 1.),
        VarData("cd", [num_nodes, num_radial], 1.),
        VarData("cn", [num_nodes, num_radial], 1.),
        VarData("ct", [num_nodes, num_radial], 1.),
        VarData("F", [num_nodes, num_radial], 1.),
        VarData("G", [num_nodes, num_radial], 1.)]

    partials_data = Array{PartialsData, 1}()

    rows, cols = get_rows_cols(
        of_shape=(num_nodes, num_radial), of_ss="ij",
        wrt_shape=(1,), wrt_ss="k")
    of_names = ["phi", "Np", "Tp", "a", "ap", "u", "v", "alpha", "W", "cl", "cd", "cn", "ct", "F", "G"]
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
        push!(partials_data, PartialsData(name, "pitch", rows=rows, cols=cols))
    end

    rows, cols = get_rows_cols(
        of_shape=(num_nodes, num_radial), of_ss="ij",
        wrt_shape=(num_nodes, num_radial), wrt_ss="ij")
    for name in of_names
        push!(partials_data, PartialsData(name, "Vx", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "Vy", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "phi", rows=rows, cols=cols))
    end

    for name in ["Np", "Tp", "a", "ap", "u", "v", "alpha", "W", "cl", "cd", "cn", "ct", "F", "G"]
        push!(partials_data, PartialsData(name, name, rows=rows, cols=cols, val=1.0))
    end

    return CCBladeResidualComp(num_nodes, num_radial, af, B, turbine, debug_print, input_data, output_data, partials_data)
end

function OpenMDAO.apply_nonlinear!(self::CCBladeResidualComp, inputs, outputs, residuals)
    num_nodes = self.num_nodes
    num_radial = self.num_radial

    # Outputs.
    n = 1
    out = Outputs(
        outputs["Np"][n, :],
        outputs["Tp"][n, :],
        outputs["a"][n, :],
        outputs["ap"][n, :],
        outputs["u"][n, :],
        outputs["v"][n, :],
        outputs["phi"][n, :],
        outputs["alpha"][n, :],
        outputs["W"][n, :],
        outputs["cl"][n, :],
        outputs["cd"][n, :],
        outputs["cn"][n, :],
        outputs["ct"][n, :],
        outputs["F"][n, :],
        outputs["G"][n, :])

    phis = outputs["phi"]

    for n in 1:num_nodes

        # Rotor parameters.
        rotor = Rotor(
            inputs["r"][n, :],
            inputs["chord"][n, :],
            inputs["theta"][n, :],
            fill(self.af, num_radial),
            inputs["Rhub"][1],
            inputs["Rtip"][1],
            self.B,
            self.turbine,
            inputs["precone"][1])

        # Operating point parameters.
        op_point = OperatingPoint(
             inputs["Vx"][n, :],
             inputs["Vy"][n, :],
             inputs["pitch"][n],
             inputs["rho"][n],
             inputs["mu"][n],
             inputs["asound"][n])

        for idx in 1:num_radial
            residuals["phi"][n, idx] = residual!(phis[n, idx], rotor, op_point, out, idx, true)
        end

        @. residuals["Np"][n, :] = outputs["Np"][n, :] - out.Np
        @. residuals["Tp"][n, :] = outputs["Tp"][n, :] - out.Tp
        @. residuals["a"][n, :] = outputs["a"][n, :] - out.a
        @. residuals["ap"][n, :] = outputs["ap"][n, :] - out.ap
        @. residuals["u"][n, :] = outputs["u"][n, :] - out.u
        @. residuals["v"][n, :] = outputs["v"][n, :] - out.v
        @. residuals["alpha"][n, :] = outputs["alpha"][n, :] - out.alpha
        @. residuals["W"][n, :] = outputs["W"][n, :] - out.W
        @. residuals["cl"][n, :] = outputs["cl"][n, :] - out.cl
        @. residuals["cd"][n, :] = outputs["cd"][n, :] - out.cd
        @. residuals["cn"][n, :] = outputs["cn"][n, :] - out.cn
        @. residuals["ct"][n, :] = outputs["ct"][n, :] - out.ct
        @. residuals["F"][n, :] = outputs["F"][n, :] - out.F
        @. residuals["G"][n, :] = outputs["G"][n, :] - out.G

    end

end

function OpenMDAO.linearize!(self::CCBladeResidualComp, inputs, outputs, partials)
    num_nodes = self.num_nodes
    num_radial = self.num_radial

    for n in 1:num_nodes

        # Rotor parameters.
        rotor = Rotor(
            inputs["r"][n, :],
            inputs["chord"][n, :],
            inputs["theta"][n, :],
            fill(self.af, num_radial),
            inputs["Rhub"][1],
            inputs["Rtip"][1],
            self.B,
            self.turbine,
            inputs["precone"][1])

        # Operating point parameters.
        op_point = OperatingPoint(
             inputs["Vx"][n, :],
             inputs["Vy"][n, :],
             inputs["pitch"][n],
             inputs["rho"][n],
             inputs["mu"][n],
             inputs["asound"][n])

        for idx in 1:num_radial
            phi = outputs["phi"][n, idx]
            derivs = residual_partials(phi, rotor, op_point, outputs, idx)
            for (k, v) in derivs
                p = reshape(partials["phi", String(k)], num_nodes, num_radial)
                p[n, idx] = v
            end
            
            derivs = output_partials(phi, rotor, op_point, outputs, idx)
            for ((of, wrt), v) in derivs
                p = reshape(partials[String(of), String(wrt)], num_nodes, num_radial)
                p[n, idx] = -v
            end
        end
    end

end

function OpenMDAO.solve_nonlinear!(self::CCBladeResidualComp, inputs, outputs)
    num_nodes = self.num_nodes
    num_radial = self.num_radial

    for n in 1:num_nodes

        # Rotor parameters.
        rotor = Rotor(
            inputs["r"][n, :],
            inputs["chord"][n, :],
            inputs["theta"][n, :],
            fill(self.af, num_radial),
            inputs["Rhub"][1],
            inputs["Rtip"][1],
            self.B,
            self.turbine,
            inputs["precone"][1])

        # Operating point parameters.
        op_point = OperatingPoint(
             inputs["Vx"][n, :],
             inputs["Vy"][n, :],
             inputs["pitch"][n],
             inputs["rho"][n],
             inputs["mu"][n],
             inputs["asound"][n])

        # Do it.
        out = solve(rotor, op_point)

        # Set the outputs.
        @. outputs["phi"][n, :] = out.phi
        @. outputs["Np"][n, :] = out.Np
        @. outputs["Tp"][n, :] = out.Tp
        @. outputs["a"][n, :] = out.a
        @. outputs["ap"][n, :] = out.ap
        @. outputs["u"][n, :] = out.u
        @. outputs["v"][n, :] = out.v
        @. outputs["alpha"][n, :] = out.alpha
        @. outputs["W"][n, :] = out.W
        @. outputs["cl"][n, :] = out.cl
        @. outputs["cd"][n, :] = out.cd
        @. outputs["cn"][n, :] = out.cn
        @. outputs["ct"][n, :] = out.ct
        @. outputs["F"][n, :] = out.F
        @. outputs["G"][n, :] = out.G

    end

end
