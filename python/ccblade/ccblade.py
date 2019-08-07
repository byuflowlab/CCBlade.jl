import numpy as np
from openmdao.api import (ImplicitComponent, ExplicitComponent, Group,
                          AkimaSplineComp, ExecComp, NewtonSolver,
                          DirectSolver, BoundsEnforceLS)
import julia.Main as jlmain


def get_rows_cols(of_shape, of_ss, wrt_shape, wrt_ss):

    if len(of_shape) != len(of_ss):
        msg = "length of of_shape {} and of_ss {} should match".format(
            of_shape, of_ss)
        raise ValueError(msg)

    if len(wrt_shape) != len(wrt_ss):
        msg = "length of wrt_shape {} and wrt_ss {} should match".format(
            wrt_shape, wrt_ss)
        raise ValueError(msg)

    # Get the output subscript, which will be an alphabetical list of
    # all the input subscripts (e.g., of_ss='ij', wrt_ss='jk' would
    # result in 'ijk'). Needed to disable the implied sumation over
    # repeated indices.
    ss = "".join(sorted(set(of_ss + wrt_ss)))
    ss = of_ss + ',' + wrt_ss + '->' + ss

    # Shamelessly stolen from John Hwang's OpenBEMT code.
    a = np.arange(np.prod(of_shape)).reshape(of_shape)
    b = np.ones(wrt_shape, dtype=int)
    rows = np.einsum(ss, a, b).flatten()

    a = np.ones(of_shape, dtype=int)
    b = np.arange(np.prod(wrt_shape)).reshape(wrt_shape)
    cols = np.einsum(ss, a, b).flatten()

    return rows, cols


class CCBladeResidualComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('B', types=int)
        self.options.declare('af_fname', types=str)
        self.options.declare('turbine', types=bool)
        self.options.declare('debug_print', types=bool, default=False)

        jlmain.eval("""
            using CCBlade
            using PyCall

            function residuals_kernel!(options, inputs, outputs, residuals)
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
                # @show residuals["phi"]

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

            residuals! = pyfunction(residuals_kernel!,
                         PyDict{String, PyAny},
                         PyDict{String, PyArray},
                         PyDict{String, PyArray},
                         PyDict{String, PyArray})

            function residual_partials_kernel!(options, inputs, outputs, partials)
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

                # Phi, the implicit variable.
                phis = outputs["phi"]

                # Get the derivatives of the residual.
                residual_derivs = CCBlade.residual_partials.(phis, sections, inflows, rotor)

                @. partials["phi", "phi"] = getfield(residual_derivs, :phi)
                @. partials["phi", "radii"] = getfield(residual_derivs, :r)
                @. partials["phi", "chord"] = getfield(residual_derivs, :chord)
                @. partials["phi", "theta"] = getfield(residual_derivs, :theta)
                @. partials["phi", "Vx"] = getfield(residual_derivs, :Vx)
                @. partials["phi", "Vy"] = getfield(residual_derivs, :Vy)
                @. partials["phi", "rho"] = getfield(residual_derivs, :rho)
                @. partials["phi", "mu"] = getfield(residual_derivs, :mu)
                @. partials["phi", "asound"] = getfield(residual_derivs, :asound)
                @. partials["phi", "hub_radius"] = getfield(residual_derivs, :Rhub)
                @. partials["phi", "prop_radius"] = getfield(residual_derivs, :Rtip)
                @. partials["phi", "precone"] = getfield(residual_derivs, :precone)

                # Get the derivatives of the outputs.
                output_derivs = CCBlade.output_partials.(phis, sections, inflows, rotor)

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
                            @. partials[of_name, wrt_name] = -getindex(output_derivs, of_idx, wrt_idx)
                        end
                    end
                end

            end

            residual_partials! = pyfunction(
                residual_partials_kernel!,
                PyDict{String, PyAny}, # options
                PyDict{String, PyArray}, # inputs
                PyDict{String, PyArray}, # outputs
                PyDict{Tuple{String, String}, PyArray}) # partials

            function guess_nonlinear_kernel!(options, inputs, phi_1, phi_2)
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
                success = @. getindex(out, 1)
                if ! all(success)
                    # error("Bracketing failed")
                    println("CCBlade bracketing failed")
                    @warn "CCBlade bracketing failed"
                end

                # Copy the left and right value of each interval.
                @. phi_1 = getindex(out, 2)
                @. phi_2 = getindex(out, 3)

            end

            guess_nonlinear! = pyfunction(
                guess_nonlinear_kernel!,
                PyDict{String, PyAny}, # options
                PyDict{String, PyArray}, # inputs
                PyArray, # phi_1
                PyArray) # phi_2


        """)
        self._julia_apply_nonlinear = jlmain.residuals_b
        self._julia_linearize = jlmain.residual_partials_b
        self._julia_guess_nonlinear = jlmain.guess_nonlinear_b

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        # Create the airfoil interpolation function. Is this the best way to do
        # this? I don't know.
        af_fname = self.options['af_fname']
        jlmain.eval(f"""
            using CCBlade
            airfoil_interp = af_from_file("{af_fname}", use_interpolations_jl=true)
        """)
        self._airfoil_interp = jlmain.airfoil_interp

        self.add_input('radii', shape=(1, num_radial), units='m')
        self.add_input('chord', shape=(1, num_radial), units='m')
        self.add_input('theta', shape=(1, num_radial), units='rad')
        self.add_input('Vx', shape=(num_nodes, 1), units='m/s')
        self.add_input('Vy', shape=(num_nodes, num_radial), units='m/s')
        self.add_input('rho', shape=(num_nodes, 1), units='kg/m**3')
        self.add_input('mu', shape=(num_nodes, 1), units='N/m**2*s')
        self.add_input('asound', shape=(num_nodes, 1), units='m/s')
        self.add_input('hub_radius', shape=(1, 1), units='m')
        self.add_input('prop_radius', shape=(1, 1), units='m')
        self.add_input('precone', shape=(1, 1), units='rad')

        self.add_output('phi', shape=(num_nodes, num_radial), units='rad')
        self.add_output('Np', shape=(num_nodes, num_radial), units='N/m')
        self.add_output('Tp', shape=(num_nodes, num_radial), units='N/m')
        self.add_output('a', shape=(num_nodes, num_radial))
        self.add_output('ap', shape=(num_nodes, num_radial))
        self.add_output('u', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('v', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('W', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('cl', shape=(num_nodes, num_radial))
        self.add_output('cd', shape=(num_nodes, num_radial))
        self.add_output('F', shape=(num_nodes, num_radial))

        # Names of all the 2D outputs.
        of_names = ('phi', 'Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd',
                    'F')

        rows, cols = get_rows_cols(
            of_shape=(num_nodes, num_radial), of_ss='ij',
            wrt_shape=(1,), wrt_ss='k')
        for name in of_names:
            self.declare_partials(name, 'hub_radius', rows=rows, cols=cols)
            self.declare_partials(name, 'prop_radius', rows=rows, cols=cols)
            self.declare_partials(name, 'precone', rows=rows, cols=cols)

        rows, cols = get_rows_cols(
            of_shape=(num_nodes, num_radial), of_ss='ij',
            wrt_shape=(num_radial,), wrt_ss='j')
        for name in of_names:
            self.declare_partials(name, 'radii', rows=rows, cols=cols)
            self.declare_partials(name, 'chord', rows=rows, cols=cols)
            self.declare_partials(name, 'theta', rows=rows, cols=cols)

        rows, cols = get_rows_cols(
            of_shape=(num_nodes, num_radial), of_ss='ij',
            wrt_shape=(num_nodes,), wrt_ss='i')
        for name in of_names:
            self.declare_partials(name, 'Vx', rows=rows, cols=cols)
            self.declare_partials(name, 'rho', rows=rows, cols=cols)
            self.declare_partials(name, 'mu', rows=rows, cols=cols)
            self.declare_partials(name, 'asound', rows=rows, cols=cols)

        rows, cols = get_rows_cols(
            of_shape=(num_nodes, num_radial), of_ss='ij',
            wrt_shape=(num_nodes, num_radial), wrt_ss='ij')
        for name in of_names:
            self.declare_partials(name, 'Vy', rows=rows, cols=cols)
            self.declare_partials(name, 'phi', rows=rows, cols=cols)

        # For the explicit outputs, the derivatives wrt themselves are
        # constant.
        for name in ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd', 'F'):
            self.declare_partials(name, name, rows=rows, cols=cols, val=1.)

    def apply_nonlinear(self, inputs, outputs, residuals):
        # options_d = dict(self.options)
        options_d = {
            'B': self.options['B'],
            'af': self._airfoil_interp,
            'turbine': self.options['turbine']}
        inputs_d = dict(inputs)
        outputs_d = dict(outputs)
        residuals_d = dict(residuals)
        self._julia_apply_nonlinear(
            options_d, inputs_d, outputs_d, residuals_d)

    def linearize(self, inputs, outputs, partials):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        # options_d = dict(self.options)
        options_d = {
            'B': self.options['B'],
            'af': self._airfoil_interp,
            'turbine': self.options['turbine']}
        inputs_d = dict(inputs)
        outputs_d = dict(outputs)
        # partials_d = dict(partials)
        partials_d = {}
        of_names = ('phi', 'Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd',
                    'F')
        wrt_names = ('radii', 'chord', 'theta', 'Vx', 'Vy', 'rho', 'mu',
                     'asound', 'hub_radius', 'prop_radius', 'precone', 'phi')
        for of_name in of_names:
            for wrt_name in wrt_names:
                partials_d[of_name, wrt_name] = partials[of_name, wrt_name]
                partials_d[of_name, wrt_name].shape = (num_nodes, num_radial)

        self._julia_linearize(options_d, inputs_d, outputs_d, partials_d)

        for of_name in of_names:
            for wrt_name in wrt_names:
                partials_d[of_name, wrt_name].shape = (-1,)

    def recurse_solve(self):
        for ind, sub in enumerate(self._subsystems_myproc):
                isub = self._subsystems_myproc_inds[ind]
                self._transfer('nonlinear', 'fwd', isub)
                sub._solve_nonlinear()
        self._apply_nonlinear()

    def guess_nonlinear(self, inputs, outputs, residuals):
        DEBUG_PRINT = self.options['debug_print']

        # I think this steps though and runs compute/apply_nonlinear to any
        # sub-components.
        self.recurse_solve()

        # If the residual norm is small, we're close enough, so return.
        GUESS_TOL = 1e-4
        res_norm = residuals.get_norm()
        if res_norm < GUESS_TOL:
            out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd', 'F')
            for name in out_names:
                if np.all(np.logical_not(np.isnan(residuals[name]))):
                    outputs[name] -= residuals[name]
            if DEBUG_PRINT:
                print(
                    f"guess_nonlinear res_norm: {res_norm} (skipping guess_nonlinear)")
            return

        options_d = {
            'B': self.options['B'],
            'af': self._airfoil_interp,
            'turbine': self.options['turbine']}
        inputs_d = dict(inputs)

        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        phi_1 = np.zeros((num_nodes, num_radial))
        phi_2 = np.zeros((num_nodes, num_radial))

        # Get an interval that brackets each root.
        self._julia_guess_nonlinear(options_d, inputs_d, phi_1, phi_2)

        # Initialize the residuals.
        res_1 = np.zeros_like(phi_1)
        outputs['phi'][:, :] = phi_1
        self.recurse_solve()
        res_1[:, :] = residuals['phi']

        res_2 = np.zeros_like(phi_1)
        outputs['phi'][:, :] = phi_2
        self.recurse_solve()
        res_2[:, :] = residuals['phi']

        # now initialize the phi_1 and phi_2 vectors so they represent the correct brackets
        mask = res_1 > 0.
        phi_1[mask], phi_2[mask] = phi_2[mask], phi_1[mask]
        res_1[mask], res_2[mask] = res_2[mask], res_1[mask]

        if DEBUG_PRINT:
            print("0 Still bracking a root?", np.all(res_1*res_2 < 0.))

        for i in range(100):
            outputs['phi'][:] = 0.5 * (phi_1 + phi_2)
            self.recurse_solve()
            new_res = residuals['phi']
            # print('iter', i, np.linalg.norm(new_res))

            # only need to do this to get into the ballpark
            res_norm = np.linalg.norm(new_res)
            if res_norm < GUESS_TOL:
                out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd',
                             'F')
                for name in out_names:
                    if np.all(np.logical_not(np.isnan(residuals[name]))):
                        outputs[name] -= residuals[name]
                if DEBUG_PRINT:
                    print(
                        f"guess_nonlinear res_norm: {res_norm}, convergence criteria satisfied")
                    print(f"Vy/r = {inputs['Vy']/inputs['radii']}")
                break

            mask_1 = new_res < 0
            mask_2 = new_res > 0

            phi_1[mask_1] = outputs['phi'][mask_1]
            res_1[mask_1] = new_res[mask_1]

            phi_2[mask_2] = outputs['phi'][mask_2]
            res_2[mask_2] = new_res[mask_2]

            if DEBUG_PRINT:
                print(f"{i+1} res_norm = {res_norm}, Still bracking a root?", np.all(res_1*res_2 < 0.))

        else:
            out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd', 'F')
            for name in out_names:
                if np.all(np.logical_not(np.isnan(residuals[name]))):
                    outputs[name] -= residuals[name]
            if DEBUG_PRINT:
                print(f"guess_nonlinear res_norm = {res_norm} > GUESS_TOL")


class CCBladeThrustTorqueComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('B', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_input('radii', shape=num_radial, units='m')
        self.add_input('dradii', shape=num_radial, units='m')
        self.add_input('Np',
                       shape=(num_nodes, num_radial), units='N/m')
        self.add_input('Tp',
                       shape=(num_nodes, num_radial), units='N/m')
        self.add_output('thrust', shape=num_nodes, units='N')
        self.add_output('torque', shape=num_nodes, units='N*m')

        rows, cols = get_rows_cols(
            of_shape=(num_nodes,), of_ss='i',
            wrt_shape=(num_radial,), wrt_ss='j')
        self.declare_partials('thrust', 'dradii', rows=rows, cols=cols)
        self.declare_partials('torque', 'dradii', rows=rows, cols=cols)
        self.declare_partials('torque', 'radii', rows=rows, cols=cols)

        self.declare_partials('thrust', 'Np', rows=rows, cols=cols)
        self.declare_partials('torque', 'Tp', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        B = self.options['B']
        radii = inputs['radii'][np.newaxis, :]
        dradii = inputs['dradii'][np.newaxis, :]
        Np = inputs['Np']
        Tp = inputs['Tp']

        outputs['thrust'][:] = B*np.sum(Np * dradii, axis=1)
        outputs['torque'][:] = B*np.sum(Tp * radii * dradii, axis=1)

    def compute_partials(self, inputs, partials):
        B = self.options['B']
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        radii = inputs['radii'][np.newaxis, :]
        dradii = inputs['dradii'][np.newaxis, :]
        Np = inputs['Np']
        Tp = inputs['Tp']

        deriv = partials['thrust', 'dradii']
        deriv.shape = (num_nodes, num_radial)
        deriv[:, :] = B * Np
        deriv.shape = (-1,)

        deriv = partials['thrust', 'Np']
        deriv.shape = (num_nodes, num_radial)
        deriv[:, :] = B * dradii
        deriv.shape = (-1,)

        deriv = partials['torque', 'dradii']
        deriv.shape = (num_nodes, num_radial)
        deriv[:, :] = B * Tp * radii
        deriv.shape = (-1,)

        deriv = partials['torque', 'radii']
        deriv.shape = (num_nodes, num_radial)
        deriv[:, :] = B * Tp * dradii
        deriv.shape = (-1,)

        deriv = partials['torque', 'Tp']
        deriv.shape = (num_nodes, num_radial)
        deriv[:, :] = B * radii * dradii
        deriv.shape = (-1,)


class CCBladeGroup(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        # self.options.declare('num_cp', types=int)
        self.options.declare('num_blades', types=int)
        self.options.declare('af_filename', types=str)
        # self.options.declare('turbine', types=bool)

    def setup(self):
        num_nodes = self.options['num_nodes']
        # num_cp = self.options['num_cp']
        num_radial = self.options['num_radial']
        num_blades = self.options['num_blades']
        af_filename = self.options['af_filename']

        comp = ExecComp('hub_radius = 0.5*hub_diameter',
                        hub_radius={'value': 0.1, 'units': 'm'},
                        hub_diameter={'units': 'm'})
        self.add_subsystem('hub_radius_comp', comp, promotes=['*'])

        comp = ExecComp('prop_radius = 0.5*prop_diameter',
                        prop_radius={'value': 1.0, 'units': 'm'},
                        prop_diameter={'units': 'm'})
        self.add_subsystem('prop_radius_comp', comp, promotes=['*'])

        comp = ExecComp('Vx = v*cos(precone)',
                        v={'units': 'm/s', 'shape': num_nodes},
                        precone={'units': 'rad'},
                        Vx={'units': 'm/s', 'shape': num_nodes})
        self.add_subsystem('Vx_comp', comp, promotes=['*'])

        comp = ExecComp('Vy = omega*radii*cos(precone)',
                        omega={'units': 'rad/s',
                               'shape': (num_nodes, 1),
                               'flat_src_indices': [0]},
                        radii={'units': 'm', 'shape': (1, num_radial)},
                        precone={'units': 'rad'},
                        Vy={'units': 'm/s', 'shape': (num_nodes, num_radial)})
        self.add_subsystem('Vy_comp', comp, promotes=['*'])

        comp = CCBladeResidualComp(num_nodes=num_nodes, num_radial=num_radial,
                                   B=num_blades, turbine=False,
                                   af_fname=af_filename, debug_print=True)
        comp.nonlinear_solver = NewtonSolver()
        comp.nonlinear_solver.options['solve_subsystems'] = True
        comp.nonlinear_solver.options['iprint'] = 2
        comp.nonlinear_solver.options['maxiter'] = 30
        comp.nonlinear_solver.options['err_on_non_converge'] = True
        comp.nonlinear_solver.options['atol'] = 1e-5
        comp.nonlinear_solver.options['rtol'] = 1e-8
        comp.nonlinear_solver.linesearch = BoundsEnforceLS()
        comp.linear_solver = DirectSolver(assemble_jac=True)
        self.add_subsystem('ccblade_comp', comp,
                           promotes_inputs=['radii', 'chord', 'theta', 'Vx',
                                            'Vy', 'rho', 'mu', 'asound',
                                            'hub_radius', 'prop_radius',
                                            'precone'],
                           promotes_outputs=['Np', 'Tp'])

        comp = CCBladeThrustTorqueComp(
            num_nodes=num_nodes, num_radial=num_radial, B=num_blades)
        self.add_subsystem('ccblade_torquethrust_comp', comp,
                           promotes_inputs=['radii', 'dradii', 'Np', 'Tp'],
                           promotes_outputs=['thrust', 'torque'])

        comp = ExecComp('efficiency = (thrust*v)/(torque*omega)',
                        thrust={'units': 'N', 'shape': num_nodes},
                        v={'units': 'm/s', 'shape': num_nodes},
                        torque={'units': 'N*m', 'shape': num_nodes},
                        omega={'units': 'rad/s', 'shape': num_nodes},
                        efficiency={'shape': num_nodes})
        self.add_subsystem('efficiency_comp', comp,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.linear_solver = DirectSolver(assemble_jac=True)

