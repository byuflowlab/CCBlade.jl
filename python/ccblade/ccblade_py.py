import numpy as np

import openmdao.api as om


def dummy_airfoil(alpha, Re, Mach):
    cl = 2*np.pi*alpha
    cd = np.zeros_like(cl)

    return cl, cd


def abs_cs(x):
    return np.sqrt(x*x)


class CCBladeResidualComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('airfoil_interp')
        self.options.declare('turbine', types=bool)
        self.options.declare('debug_print', types=bool, default=False)
        self.options.declare('solve_nonlinear', types=bool, default=True)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_discrete_input('B', val=3)
        self.add_input('radii', shape=(1, num_radial), units='m')
        self.add_input('chord', shape=(1, num_radial), units='m')
        self.add_input('theta', shape=(1, num_radial), units='rad')
        self.add_input('Vx', shape=(num_nodes, num_radial), units='m/s')
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

        self.declare_partials('*', '*')

        deriv_method = 'fd'
        self.declare_coloring(wrt='*', method=deriv_method, perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def apply_nonlinear(self, inputs, outputs, residuals,
                        discrete_inputs, discrete_outputs):

        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        turbine = self.options['turbine']
        B = discrete_inputs['B']
        af = self.options['airfoil_interp']

        r = inputs['radii']
        chord = inputs['chord']
        theta = inputs['theta']
        Vx = inputs['Vx']
        Vy = inputs['Vy']
        rho = inputs['rho']
        mu = inputs['mu']
        asound = inputs['asound']
        Rhub = inputs['hub_radius']
        Rtip = inputs['prop_radius']

        phi = outputs['phi']

        # check if turbine or propeller and change input sign if necessary
        # swapsign = turbine ? 1 : -1
        swapsign = 1 if turbine else -1
        # Bad idea:
        # theta *= swapsign
        # Vx *= swapsign
        theta = swapsign * inputs['theta']
        Vx = swapsign * inputs['Vx']

        # constants
        sigma_p = B*chord/(2.0*np.pi*r)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        # angle of attack
        alpha = phi - theta

        # Reynolds number
        W0 = np.sqrt(Vx*Vx + Vy*Vy)  # ignoring induction, which is generally a very minor error and only affects Reynolds number
        Re = rho * W0 * chord / mu

        # Mach number
        Mach = W0/asound  # also ignoring induction

        # airfoil cl/cd
        cl, cd = af(alpha, Re, Mach)

        # resolve into normal and tangential forces
        cn = cl*cphi + cd*sphi
        ct = cl*sphi - cd*cphi

        # Prandtl's tip and hub loss factor
        factortip = B/2.0*(Rtip - r)/(r*abs_cs(sphi))
        Ftip = 2.0/np.pi*np.arccos(np.exp(-factortip))
        factorhub = B/2.0*(r - Rhub)/(Rhub*abs_cs(sphi))
        Fhub = 2.0/np.pi*np.arccos(np.exp(-factorhub))
        F = Ftip * Fhub

        # sec parameters
        k = cn*sigma_p/(4.0*F*sphi*sphi)
        kp = ct*sigma_p/(4.0*F*sphi*cphi)

        # if phi < 0:
        #     k = -k
        k[phi < 0.] *= -1.

        # if k <= 2.0/3:  # momentum region
        #     a = k/(1 + k)

        # else:  # empirical region
        #     g1 = 2.0*F*k - (10.0/9-F)
        #     g2 = 2.0*F*k - (4.0/3-F)*F
        #     g3 = 2.0*F*k - (25.0/9-2*F)

        #     if np.isclose(g3, 0.0, atol=1e-6):  # avoid singularity
        #         a = 1.0 - 1.0/(2.0*np.sqrt(g2))
        #     else:
        #         a = (g1 - np.sqrt(g2)) / g3

        a = np.zeros_like(phi)
        for i in range(num_nodes):
            for j in range(num_radial):
                if k[i, j] <= 2./3.:
                    a[i, j] = k[i, j]/(1 + k[i, j])
                else:
                    g1 = 2.0*F[i, j]*k[i, j] - (10.0/9-F[i, j])
                    g2 = 2.0*F[i, j]*k[i, j] - (4.0/3-F[i, j])*F[i, j]
                    g3 = 2.0*F[i, j]*k[i, j] - (25.0/9-2*F[i, j])
                    if np.isclose(g3, 0.0, atol=1e-6):
                        a[i, j] = 1.0 - 1.0/(2.0*np.sqrt(g2))
                    else:
                        a[i, j] = (g1 - np.sqrt(g2)) / g3

        # mom_mask = k <= 2./3.
        # a[mom_mask] = k[mom_mask]/(1 + k[mom_mask])
        # print(f"a (1) = {a}")

        # emp_mask = np.logical_not(mom_mask)
        # g1 = 2.0*F*k - (10.0/9-F)
        # g2 = 2.0*F*k - (4.0/3-F)*F
        # g3 = 2.0*F*k - (25.0/9-2*F)

        # # print(f"k = {k}")
        # # print(f"F = {F}")
        # # print(f"g2 = {g2}")
        # # sing_mask = np.logical_and(emp_mask, np.isclose(g3, 0.0, atol=1e-6))
        # sing_mask = np.isclose(g3, 0.0, atol=1e-6)
        # mask_mask = np.logical_and(emp_mask, sing_mask)
        # a[mask_mask] = 1.0 - 1.0/(2.0*np.sqrt(g2[sing_mask]))
        # # print(f"a (2) = {a}")

        # mask_mask = np.logical_and(emp_mask, np.logical_not(sing_mask))
        # a[mask_mask] = (g1[mask_mask] - np.sqrt(g2[mask_mask])) / g3[mask_mask]
        # # print(f"a (3} = {a}")

        u = a * Vx

        # -------- tangential induction ----------
        kp[Vx < 0.] *= -1.

        ap = kp/(1 - kp)
        v = ap * Vy

        # ------- residual function -------------
        residuals['phi'] = np.sin(phi)/(Vx - u) - np.cos(phi)/(Vy + v)

        # if np.isclose(kp, 1.0, atol=1e-6):  # state corresopnds to Vy=0, return any nonzero residual
        #     return 1.0, Outputs()
        residuals['phi'][np.isclose(kp, 1.0, atol=1e-6)] = 1.

        # if isapprox(k, -1.0, atol=1e-6)  # state corresopnds to Vx=0, return any nonzero residual
        #     return 1.0, Outputs()
        residuals['phi'][np.isclose(k, -1.0, atol=1e-6)] = 1.

        # ------- loads ---------
        W2 = (Vx - u)**2 + (Vy + v)**2
        residuals['Np'] = cn*0.5*rho*W2*chord - outputs['Np']
        residuals['Tp'] = ct*0.5*rho*W2*chord*swapsign - outputs['Tp']

        # Other residuals.
        residuals['a'] = a - outputs['a']
        residuals['ap'] = ap - outputs['ap']
        residuals['u'] = u - outputs['u']
        residuals['v'] = v*swapsign - outputs['v']
        residuals['W'] = np.sqrt(W2) - outputs['W']
        residuals['cl'] = cl - outputs['cl']
        residuals['cd'] = cl - outputs['cd']
        residuals['F'] = F - outputs['F']

    def solve_nonlinear(self, inputs, outputs,
                        discrete_inputs, discrete_outputs):
        if not self.options['solve_nonlinear']:
            return

        turbine = self.options['turbine']
        DEBUG_PRINT = self.options['debug_print']
        residuals = self._residuals

        self.apply_nonlinear(inputs, outputs, residuals, discrete_inputs,
                             discrete_outputs)

        # If the residual norm is small, we're close enough, so return.
        GUESS_TOL = 1e-6
        res_norm = residuals.get_norm()
        if res_norm < GUESS_TOL:
            out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd', 'F')
            for name in out_names:
                if np.all(np.logical_not(np.isnan(residuals[name]))):
                    outputs[name] += residuals[name]
            if DEBUG_PRINT:
                print(
                    f"guess_nonlinear res_norm: {res_norm} (skipping guess_nonlinear)")
            return

        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        # This needs to be fixed—need to check the sign of Vx and Vy, I think.
        eps = 1.0*np.pi/180.0
        if turbine:
            phi_1 = np.zeros((num_nodes, num_radial)) + eps
            phi_2 = 0.5*np.pi*np.ones((num_nodes, num_radial)) - eps
        else:
            phi_1 = -0.5*np.pi*np.ones((num_nodes, num_radial)) + eps
            phi_2 = np.zeros((num_nodes, num_radial)) - eps

        # Initialize the residuals.
        res_1 = np.zeros_like(phi_1)
        outputs['phi'][:, :] = phi_1
        self.apply_nonlinear(inputs, outputs, residuals, discrete_inputs,
                             discrete_outputs)
        res_1[:, :] = residuals['phi']

        res_2 = np.zeros_like(phi_1)
        outputs['phi'][:, :] = phi_2
        self.apply_nonlinear(inputs, outputs, residuals, discrete_inputs,
                             discrete_outputs)
        res_2[:, :] = residuals['phi']

        # now initialize the phi_1 and phi_2 vectors so they represent the correct brackets
        mask = res_1 > 0.
        phi_1[mask], phi_2[mask] = phi_2[mask], phi_1[mask]
        res_1[mask], res_2[mask] = res_2[mask], res_1[mask]

        if DEBUG_PRINT:
            print("0 Still bracking a root?", np.all(res_1*res_2 < 0.))

        for i in range(100):
            outputs['phi'][:] = 0.5 * (phi_1 + phi_2)
            self.apply_nonlinear(inputs, outputs, residuals, discrete_inputs,
                                 discrete_outputs)
            new_res = residuals['phi']
            # print('iter', i, np.linalg.norm(new_res))

            # only need to do this to get into the ballpark
            res_norm = np.linalg.norm(new_res)
            if res_norm < GUESS_TOL:
                out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd',
                             'F')
                for name in out_names:
                    if np.all(np.logical_not(np.isnan(residuals[name]))):
                        outputs[name] += residuals[name]
                if DEBUG_PRINT:
                    print(
                        f"guess_nonlinear res_norm: {res_norm}, convergence criteria satisfied")
                break

            mask_1 = new_res < 0
            mask_2 = new_res > 0

            phi_1[mask_1] = outputs['phi'][mask_1]
            res_1[mask_1] = new_res[mask_1]

            phi_2[mask_2] = outputs['phi'][mask_2]
            res_2[mask_2] = new_res[mask_2]

        else:
            out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd', 'F')
            for name in out_names:
                if np.all(np.logical_not(np.isnan(residuals[name]))):
                    outputs[name] += residuals[name]
            if DEBUG_PRINT:
                print(f"guess_nonlinear res_norm = {res_norm} > GUESS_TOL")


class CCBladeThrustTorqueComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_discrete_input('B', val=3)
        self.add_input('radii', shape=num_radial, units='m')
        self.add_input('dradii', shape=num_radial, units='m')
        self.add_input('Np',
                       shape=(num_nodes, num_radial), units='N/m')
        self.add_input('Tp',
                       shape=(num_nodes, num_radial), units='N/m')
        self.add_output('thrust', shape=num_nodes, units='N')
        self.add_output('torque', shape=num_nodes, units='N*m')

        self.declare_partials('thrust', 'dradii')
        self.declare_partials('torque', 'dradii')
        self.declare_partials('torque', 'radii')
        self.declare_partials('thrust', 'Np')
        self.declare_partials('torque', 'Tp')

        # turn on dynamic partial coloring
        self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        B = discrete_inputs['B']
        radii = inputs['radii'][np.newaxis, :]
        dradii = inputs['dradii'][np.newaxis, :]
        Np = inputs['Np']
        Tp = inputs['Tp']

        outputs['thrust'][:] = B*np.sum(Np * dradii, axis=1)
        outputs['torque'][:] = B*np.sum(Tp * radii * dradii, axis=1)


class CCBladeGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('airfoil_interp')
        self.options.declare('turbine', types=bool)
        self.options.declare('phi_residual_solve_nonlinear', types=bool,
                             default=True)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        airfoil_interp = self.options['airfoil_interp']
        turbine = self.options['turbine']
        solve_nonlinear = self.options['phi_residual_solve_nonlinear']

        comp = om.ExecComp('hub_radius = 0.5*hub_diameter',
                           hub_radius={'value': 0.1, 'units': 'm'},
                           hub_diameter={'units': 'm'})
        self.add_subsystem('hub_radius_comp', comp, promotes=['*'])

        comp = om.ExecComp('prop_radius = 0.5*prop_diameter',
                           prop_radius={'value': 1.0, 'units': 'm'},
                           prop_diameter={'units': 'm'})
        self.add_subsystem('prop_radius_comp', comp, promotes=['*'])

        # src_indices = np.zeros((num_nodes, num_radial, 1), dtype=int)
        # comp = om.ExecComp(
        #     'Vx = v*cos(precone)',
        #     v={'units': 'm/s', 'shape': (num_nodes, num_radial),
        #        'src_indices': src_indices},
        #     precone={'units': 'rad'},
        #     Vx={'units': 'm/s', 'shape': (num_nodes, num_radial)})
        # self.add_subsystem('Vx_comp', comp, promotes=['*'])

        # comp = om.ExecComp('Vy = omega*radii*cos(precone)',
        #                    omega={'units': 'rad/s',
        #                           'shape': (num_nodes, 1),
        #                           'flat_src_indices': [0]},
        #                    radii={'units': 'm', 'shape': (1, num_radial)},
        #                    precone={'units': 'rad'},
        #                    Vy={'units': 'm/s', 'shape': (num_nodes, num_radial)})
        # self.add_subsystem('Vy_comp', comp, promotes=['*'])

        comp = CCBladeResidualComp(
            num_nodes=num_nodes, num_radial=num_radial, turbine=turbine,
            airfoil_interp=airfoil_interp, debug_print=True,
            solve_nonlinear=solve_nonlinear)
        # comp.nonlinear_solver = om.NewtonSolver()
        # comp.nonlinear_solver.options['solve_subsystems'] = True
        # comp.nonlinear_solver.options['iprint'] = 2
        # comp.nonlinear_solver.options['maxiter'] = 30
        # comp.nonlinear_solver.options['err_on_non_converge'] = True
        # comp.nonlinear_solver.options['atol'] = 1e-5
        # comp.nonlinear_solver.options['rtol'] = 1e-8
        # comp.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        self.add_subsystem('ccblade_comp', comp,
                           promotes_inputs=['B', 'radii', 'chord', 'theta',
                                            'Vx', 'Vy', 'rho', 'mu', 'asound',
                                            'hub_radius', 'prop_radius',
                                            'precone'],
                           promotes_outputs=['Np', 'Tp'])

        comp = CCBladeThrustTorqueComp(num_nodes=num_nodes,
                                       num_radial=num_radial)
        self.add_subsystem('ccblade_torquethrust_comp', comp,
                           promotes_inputs=['radii', 'dradii', 'Np', 'Tp'],
                           promotes_outputs=['thrust', 'torque'])

        comp = om.ExecComp('efficiency = (thrust*v)/(torque*omega)',
                           thrust={'units': 'N', 'shape': num_nodes},
                           v={'units': 'm/s', 'shape': num_nodes},
                           torque={'units': 'N*m', 'shape': num_nodes},
                           omega={'units': 'rad/s', 'shape': num_nodes},
                           efficiency={'shape': num_nodes})
        self.add_subsystem('efficiency_comp', comp,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.linear_solver = om.DirectSolver(assemble_jac=True)


def complicated():
    num_nodes = 1
    B = 2  # number of blades
    af = dummy_airfoil
    turbine = False

    Rhub = np.array([0.5 * 0.0254]).reshape((1, 1))
    Rtip = np.array([3.0 * 0.0254]).reshape((1, 1))
    precone = np.array([0.0]).reshape((1, 1))

    r = 0.0254*np.array(
        [0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332, 1.1128,
         1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911, 1.6708, 1.7505,
         1.8302, 1.9099, 1.9896, 2.0693, 2.1490, 2.2287, 2.3084, 2.3881,
         2.4678, 2.5475, 2.6273, 2.7070, 2.7867, 2.8661, 2.9410]).reshape(
             1, -1)
    num_radial = r.shape[-1]
    r = np.array(r).reshape((1, num_radial))

    chord = 0.0254*np.array(
        [0.6270, 0.6255, 0.6231, 0.6199, 0.6165, 0.6125, 0.6054, 0.5973,
         0.5887, 0.5794, 0.5695, 0.5590, 0.5479, 0.5362, 0.5240, 0.5111,
         0.4977, 0.4836, 0.4689, 0.4537, 0.4379, 0.4214, 0.4044, 0.3867,
         0.3685, 0.3497, 0.3303, 0.3103, 0.2897, 0.2618, 0.1920]).reshape(
             (1, num_radial))

    theta = np.pi/180.0*np.array(
        [40.2273, 38.7657, 37.3913, 36.0981, 34.8803, 33.5899, 31.6400,
         29.7730, 28.0952, 26.5833, 25.2155, 23.9736, 22.8421, 21.8075,
         20.8586, 19.9855, 19.1800, 18.4347, 17.7434, 17.1005, 16.5013,
         15.9417, 15.4179, 14.9266, 14.4650, 14.0306, 13.6210, 13.2343,
         12.8685, 12.5233, 12.2138]).reshape((1, num_radial))

    omega = np.array([8000.0*(2*np.pi/60.0)]).reshape((num_nodes, 1))

    Vinf = np.array([10.0]).reshape((num_nodes, 1))
    Vy = omega*r*np.cos(precone)
    Vx = np.tile(Vinf * np.cos(precone), (1, num_radial))

    rho = np.array([1.125]).reshape((num_nodes, 1))
    mu = np.array([1.]).reshape((num_nodes, 1))
    asound = np.array([1.]).reshape((num_nodes, 1))

    prob = om.Problem()

    comp = CCBladeResidualComp(num_nodes=num_nodes,
                               num_radial=num_radial,
                               airfoil_interp=af,
                               turbine=turbine,
                               debug_print=True)

    prob.model.add_subsystem('ccblade', comp)

    prob.setup()
    prob.final_setup()

    prob['ccblade.phi'] = -1.
    prob['ccblade.radii'] = r
    prob['ccblade.chord'] = chord
    prob['ccblade.theta'] = theta
    prob['ccblade.Vx'] = Vx
    prob['ccblade.Vy'] = Vy
    prob['ccblade.rho'] = rho
    prob['ccblade.mu'] = mu
    prob['ccblade.asound'] = asound
    prob['ccblade.hub_radius'] = Rhub
    prob['ccblade.prop_radius'] = Rtip
    prob['ccblade.precone'] = precone

    prob.run_model()
    # prob.model.run_apply_nonlinear()
    prob.model.list_outputs(residuals=True, print_arrays=True)


def simple():
    num_nodes = 1
    # B = 3  # number of blades
    precone = 0.
    af = dummy_airfoil
    turbine = True
    hub_radius = 0.01
    prop_radius = 500.0
    r = np.array([0.2, 1, 2, 3, 4, 5])[np.newaxis, :]
    num_radial = r.size

    gamma = np.array([61.0, 74.31002131, 84.89805553, 89.07195504, 91.25038415,
                      92.58003871])
    theta = ((90.0 - gamma)*np.pi/180)[np.newaxis, :]
    chord = np.array([0.7, 0.706025153, 0.436187551, 0.304517933, 0.232257636,
                      0.187279622])[np.newaxis, :]

    Vinf = np.array([7.0]).reshape((num_nodes, 1))
    tsr = 8
    omega = tsr*Vinf/5.0
    rho = 1.0

    Vx = np.tile(Vinf * np.cos(precone), (1, num_radial))
    Vy = omega*r*np.cos(precone)

    prob = om.Problem()

    comp = CCBladeResidualComp(num_nodes=num_nodes,
                               num_radial=num_radial,
                               airfoil_interp=af,
                               turbine=turbine,
                               debug_print=True)

    prob.model.add_subsystem('ccblade', comp)

    prob.setup()
    prob.final_setup()

    prob['ccblade.phi'] = np.array([0.706307, 0.416522, 0.226373, 0.153094,
                                    0.115256, 0.092305])[np.newaxis, :]
    prob['ccblade.radii'] = r
    prob['ccblade.chord'] = chord
    prob['ccblade.theta'] = theta
    prob['ccblade.Vx'] = Vx
    prob['ccblade.Vy'] = Vy
    prob['ccblade.rho'] = rho
    prob['ccblade.mu'] = 1.
    prob['ccblade.asound'] = 1.
    prob['ccblade.hub_radius'] = hub_radius
    prob['ccblade.prop_radius'] = prop_radius
    prob['ccblade.precone'] = precone

    prob.run_model()
    # prob.model.run_apply_nonlinear()
    prob.model.list_outputs(residuals=True, print_arrays=True)


if __name__ == "__main__":
    simple()
