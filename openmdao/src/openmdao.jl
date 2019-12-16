import Base.convert
using OpenMDAO
using CCBlade: solve

struct CCBladeResidualComp <: OpenMDAO.AbstractImplicitComp
    num_nodes
    num_radial
    af
    B
    turbine
    rotors
    sections
    ops
end

function CCBladeResidualComp(; num_nodes, num_radial, af, B, turbine)
    # Check if the airfoil interpolation passed is a num_radial-length array.
    try
        num_af = length(af)
        if num_af == num_radial
            af = reshape(af, 1, num_radial)
        else
            throw(DomainError("af has length $num_af, but should have length $num_radial"))
        end
    catch e
        if isa(e, MethodError)
            # af is not an array of stuff, so assume it's just a single
            # function, and make it have shape (1, num_radial).
            af = fill(af, 1, num_radial)
        else
            # Some other error happened, so rethrow it.
            rethrow(e)
        end
    end

    Rhub = fill(0., num_nodes, 1)
    Rtip = fill(1., num_nodes, 1)
    pitch = fill(0., num_nodes, 1)
    precone = fill(0., num_nodes, 1)
    rotors = Rotor.(Rhub, Rtip, B, turbine, pitch, precone)

    r = fill(1., num_nodes, num_radial)
    chord = fill(1., num_nodes, num_radial)
    theta = fill(1., num_nodes, num_radial)
    sections = Section.(r, chord, theta, af)

    Vx = fill(1., num_nodes, num_radial)
    Vy = fill(1., num_nodes, num_radial)
    rho = fill(1., num_nodes, 1)
    mu = fill(1., num_nodes, 1)
    asound = fill(1., num_nodes, 1)
    ops = OperatingPoint.(Vx, Vy, rho, mu, asound)

    return CCBladeResidualComp(num_nodes, num_radial, af, B, turbine, rotors, sections, ops)
end

function OpenMDAO.setup(self::CCBladeResidualComp)
    num_nodes = self.num_nodes
    num_radial = self.num_radial

    input_data = [
        VarData("r", shape=(num_nodes, num_radial), val=1., units="m"),
        VarData("chord", shape=(num_nodes, num_radial), val=1., units="m"),
        VarData("theta", shape=(num_nodes, num_radial), val=1., units="rad"),
        VarData("Vx", shape=(num_nodes, num_radial), val=1., units="m/s"),
        VarData("Vy", shape=(num_nodes, num_radial), val=1., units="m/s"),
        VarData("rho", shape=(num_nodes, 1), val=1., units="kg/m**3"),
        VarData("mu", shape=(num_nodes, 1), val=1., units="N/m**2*s"),
        VarData("asound", shape=(num_nodes, 1), val=1., units="m/s"),
        VarData("Rhub", shape=(num_nodes, 1), val=1., units="m"),
        VarData("Rtip", shape=(num_nodes, 1), val=1., units="m"),
        VarData("pitch", shape=(num_nodes, 1), val=1., units="rad"),
        VarData("precone", shape=(num_nodes, 1), val=1., units="rad")]

    output_data = [
        VarData("phi", shape=(num_nodes, num_radial), val=1., units="rad"),
        VarData("Np", shape=(num_nodes, num_radial), val=1., units="N/m"),
        VarData("Tp", shape=(num_nodes, num_radial), val=1., units="N/m"),
        VarData("a", shape=(num_nodes, num_radial), val=1.),
        VarData("ap", shape=(num_nodes, num_radial), val=1.),
        VarData("u", shape=(num_nodes, num_radial), val=1., units="m/s"),
        VarData("v", shape=(num_nodes, num_radial), val=1., units="m/s"),
        VarData("W", shape=(num_nodes, num_radial), val=1., units="m/s"),
        VarData("cl", shape=(num_nodes, num_radial), val=1.),
        VarData("cd", shape=(num_nodes, num_radial), val=1.),
        VarData("cn", shape=(num_nodes, num_radial), val=1.),
        VarData("ct", shape=(num_nodes, num_radial), val=1.),
        VarData("F", shape=(num_nodes, num_radial), val=1.),
        VarData("G", shape=(num_nodes, num_radial), val=1.)]

    partials_data = Vector{PartialsData}()

    of_names = ["phi", "Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "cn", "ct", "F", "G"]

    ss_sizes = Dict(:i=>num_nodes, :j=>num_radial)
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i])
    for name in of_names
        push!(partials_data, PartialsData(name, "Rhub", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "Rtip", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "pitch", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "precone", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "rho", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "mu", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "asound", rows=rows, cols=cols))
    end

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i, :j])
    for name in of_names
        push!(partials_data, PartialsData(name, "r", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "chord", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "theta", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "Vx", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "Vy", rows=rows, cols=cols))
        push!(partials_data, PartialsData(name, "phi", rows=rows, cols=cols))
    end

    for name in ["Np", "Tp", "a", "ap", "u", "v", "W", "cl", "cd", "cn", "ct", "F", "G"]
        push!(partials_data, PartialsData(name, name, rows=rows, cols=cols, val=1.0))
    end

    return input_data, output_data, partials_data
end

function OpenMDAO.linearize!(self::CCBladeResidualComp, inputs, outputs, partials)
    # Copy the data in `inputs` to the array of structs in `self`.
    set_inputs!(self, inputs)

    # Phi, the implicit variable.
    phis = outputs["phi"]

    num_nodes = self.num_nodes
    num_radial = self.num_radial

    # Get the derivatives of the residual. This will have size (num_nodes,
    # num_radial).
    residual_derivs = residual_partials.(phis, self.rotors, self.sections, self.ops)

    # Copy the derivatives of the residual to the partials dict. First do the
    # derivative of the phi residual wrt phi.
    deriv = transpose(reshape(partials["phi", "phi"], (num_radial, num_nodes)))
    @. deriv = getfield(residual_derivs, :phi)

    # Copy the derivatives of the residual wrt each input.
    # for i in self.inputs
    for str in keys(inputs)
        sym = Symbol(str)
        deriv = transpose(reshape(partials["phi", str], (num_radial, num_nodes)))
        @. deriv = getfield(residual_derivs, sym)
    end

    # Get the derivatives of the explicit outputs.
    output_derivs = output_partials.(phis, self.rotors, self.sections, self.ops)

    # Copy the derivatives of the explicit outputs into the partials dict.
    for of_str in keys(outputs)
        if of_str == "phi"
            continue
        else
            of_sym = Symbol(of_str)
            for wrt_str in keys(inputs)
                wrt_sym = Symbol(wrt_str)
                # reshape does not copy data: https://github.com/JuliaLang/julia/issues/112
                deriv = transpose(reshape(partials[of_str, wrt_str], (num_radial, num_nodes)))
                @. deriv = -getfield(getfield(output_derivs, of_sym), wrt_sym)
            end
            # Also need the derivative of each output with respect to phi, but
            # phi is an output, not an input, so we'll need to handle that
            # seperately.
            wrt_str = "phi"
            wrt_sym = Symbol(wrt_str)
            # reshape does not copy data: https://github.com/JuliaLang/julia/issues/112
            deriv = transpose(reshape(partials[of_str, wrt_str], (num_radial, num_nodes)))
            @. deriv = -getfield(getfield(output_derivs, of_sym), wrt_sym)
        end
    end

    return nothing
end

function OpenMDAO.solve_nonlinear!(self::CCBladeResidualComp, inputs, outputs)
    # Copy the data in `inputs` to the array of structs in `self`.
    set_inputs!(self, inputs)

    # When called this way, CCBlade.solve will return a 2D array of `Outputs`
    # objects.
    out = solve.(self.rotors, self.sections, self.ops)

    # Set the outputs.
    for out_str in keys(outputs)
        out_sym = Symbol(out_str)
        @. outputs[out_str] = getfield(out, out_sym)
    end

    return nothing
end

function set_inputs!(self::CCBladeResidualComp, inputs)
    # Rotor parameters.
    setfield!.(self.rotors, :Rhub, inputs["Rhub"])
    setfield!.(self.rotors, :Rtip, inputs["Rtip"])
    setfield!.(self.rotors, :precone, inputs["precone"])

    # Blade section parameters.
    setfield!.(self.sections, :r, inputs["r"])
    setfield!.(self.sections, :chord, inputs["chord"])
    setfield!.(self.sections, :theta, inputs["theta"])

    # Inflow parameters.
    setfield!.(self.ops, :Vx, inputs["Vx"])
    setfield!.(self.ops, :Vy, inputs["Vy"])
    setfield!.(self.ops, :rho, inputs["rho"])
    setfield!.(self.ops, :mu, inputs["mu"])
    setfield!.(self.ops, :asound, inputs["asound"])

    return nothing
end
