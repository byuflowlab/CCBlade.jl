import numpy as np

import openmdao.api as om
from ccblade.brentv import brentv


def dummy_airfoil(alpha, Re, Mach):
    cl = 2*np.pi*alpha
    cd = np.zeros_like(cl)

    return cl, cd


def abs_cs(x):
    return np.sqrt(x*x)


class LocalInflowAngleComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('num_blades', types=int)
        self.options.declare('airfoil_interp')
        self.options.declare('turbine', types=bool)
        self.options.declare('debug_print', types=bool, default=False)
        self.options.declare(
            'solve_nonlinear', types=str, default='brent',
            check_valid=lambda _, s: s in ['bracketing', 'brent'])

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        af = self.options['airfoil_interp']
        try:
            af[0]
        except TypeError:
            self._af = num_radial*[af]
        else:
            n_af = len(af)
            if n_af != num_radial:
                msg = f"airfoil_interp option should have length {num_radial}, but has length {n_af}"
                raise ValueError(msg)
            self._af = af

        # self.add_input('B', val=3)
        self.add_input('radii', shape=(num_nodes, num_radial), units='m')
        self.add_input('chord', shape=(num_nodes, num_radial), units='m')
        self.add_input('theta', shape=(num_nodes, num_radial), units='rad')
        self.add_input('Vx', shape=(num_nodes, num_radial), units='m/s')
        self.add_input('Vy', shape=(num_nodes, num_radial), units='m/s')
        self.add_input('rho', shape=(num_nodes, 1), units='kg/m**3')
        self.add_input('mu', shape=(num_nodes, 1), units='N/m**2*s')
        self.add_input('asound', shape=(num_nodes, 1), units='m/s')
        self.add_input('hub_radius', shape=(num_nodes, 1), units='m')
        self.add_input('prop_radius', shape=(num_nodes, 1), units='m')
        self.add_input('precone', shape=(num_nodes, 1), units='rad')

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

    def apply_nonlinear(self, inputs, outputs, residuals):

        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        B = self.options['num_blades']
        turbine = self.options['turbine']
        af = self._af

        r = inputs['radii']
        chord = inputs['chord']
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
        cl = np.zeros_like(alpha)
        cd = np.zeros_like(alpha)
        for i in range(num_radial):
            cl[:, i], cd[:, i] = af[i](alpha[:, i], Re[:, i], Mach[:, i])

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
        k[phi < 0.] *= -1.

        ####################################
        # Original Julia Code
        ####################################
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

        ####################################
        # Slow Pythonized Code
        ####################################
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

        ####################################
        # Fast Pythonized Code
        ####################################
        # mom_mask = k <= 2./3.
        # a[mom_mask] = k[mom_mask]/(1 + k[mom_mask])

        # emp_mask = np.logical_not(mom_mask)
        # g1 = 2.0*F*k - (10.0/9-F)
        # g2 = 2.0*F*k - (4.0/3-F)*F
        # g3 = 2.0*F*k - (25.0/9-2*F)

        # # sing_mask = np.logical_and(emp_mask, np.isclose(g3, 0.0, atol=1e-6))
        # sing_mask = np.isclose(g3, 0.0, atol=1e-6)
        # mask_mask = np.logical_and(emp_mask, sing_mask)
        # a[mask_mask] = 1.0 - 1.0/(2.0*np.sqrt(g2[sing_mask]))

        # mask_mask = np.logical_and(emp_mask, np.logical_not(sing_mask))
        # a[mask_mask] = (g1[mask_mask] - np.sqrt(g2[mask_mask])) / g3[mask_mask]

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
        residuals['cd'] = cd - outputs['cd']
        residuals['F'] = F - outputs['F']

    def solve_nonlinear(self, inputs, outputs):
        method = self.options['solve_nonlinear']
        if method == 'bracketing':
            self._solve_nonlinear_bracketing(
                inputs, outputs)
        elif method == 'brent':
            self._solve_nonlinear_brent(
                inputs, outputs)
        else:
            msg = f"unknown CCBlade solve_nonlinear method {method}"
            raise ValueError(msg)

    def _solve_nonlinear_bracketing(self, inputs, outputs):

        SOLVE_TOL = 1e-10
        DEBUG_PRINT = self.options['debug_print']
        residuals = self._residuals

        self.apply_nonlinear(inputs, outputs, residuals)

        bracket_found, phi_1, phi_2 = self._first_bracket(
            inputs, outputs, residuals)
        if not np.all(bracket_found):
            raise om.AnalysisError("CCBlade bracketing failed")

        # Initialize the residuals.
        res_1 = np.zeros_like(phi_1)
        outputs['phi'][:, :] = phi_1
        self.apply_nonlinear(inputs, outputs, residuals)
        res_1[:, :] = residuals['phi']

        res_2 = np.zeros_like(phi_1)
        outputs['phi'][:, :] = phi_2
        self.apply_nonlinear(inputs, outputs, residuals)
        res_2[:, :] = residuals['phi']

        # now initialize the phi_1 and phi_2 vectors so they represent the correct brackets
        mask = res_1 > 0.
        phi_1[mask], phi_2[mask] = phi_2[mask], phi_1[mask]
        res_1[mask], res_2[mask] = res_2[mask], res_1[mask]

        steps_taken = 0
        success = np.all(np.abs(phi_1 - phi_2) < SOLVE_TOL)
        while steps_taken < 50 and not success:
            outputs['phi'][:] = 0.5 * (phi_1 + phi_2)
            self.apply_nonlinear(inputs, outputs, residuals)
            new_res = residuals['phi']

            mask_1 = new_res < 0
            mask_2 = new_res > 0

            phi_1[mask_1] = outputs['phi'][mask_1]
            res_1[mask_1] = new_res[mask_1]

            phi_2[mask_2] = outputs['phi'][mask_2]
            res_2[mask_2] = new_res[mask_2]

            steps_taken += 1
            success = np.all(np.abs(phi_1 - phi_2) < SOLVE_TOL)

            # only need to do this to get into the ballpark
            if DEBUG_PRINT:
                res_norm = np.linalg.norm(new_res)
                print(f"{steps_taken} solve_nonlinear res_norm: {res_norm}")

        # Fix up the other outputs.
        out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd', 'F')
        for name in out_names:
            if np.all(np.logical_not(np.isnan(residuals[name]))):
                outputs[name] += residuals[name]

        # Fix up the other residuals.
        self.apply_nonlinear(inputs, outputs, residuals)
        if not success:
            raise om.AnalysisError(
                "CCBlade _solve_nonlinear_bracketing failed")

    def _solve_nonlinear_brent(self, inputs, outputs):
        SOLVE_TOL = 1e-10
        DEBUG_PRINT = self.options['debug_print']

        # Find brackets for the phi residual
        bracket_found, phi_1, phi_2 = self._first_bracket(
            inputs, outputs, self._residuals)
        if not np.all(bracket_found):
            raise om.AnalysisError("CCBlade bracketing failed")

        # Create a wrapper function compatible with the brentv function.
        def f(x):
            outputs['phi'][:, :] = x
            self.apply_nonlinear(inputs, outputs, self._residuals)
            return np.copy(self._residuals['phi'])

        # Find the root.
        phi, steps_taken, success = brentv(f, phi_1, phi_2, tolerance=SOLVE_TOL)

        # Fix up the other outputs.
        out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'W', 'cl', 'cd', 'F')
        for name in out_names:
            if np.all(np.logical_not(np.isnan(self._residuals[name]))):
                outputs[name] += self._residuals[name]

        # Fix up the other residuals.
        self.apply_nonlinear(inputs, outputs, self._residuals)

        if DEBUG_PRINT:
            res_norm = np.linalg.norm(self._residuals['phi'])
            print(f"CCBlade brentv steps taken: {steps_taken}, residual norm = {res_norm}")

        if not success:
            raise om.AnalysisError(
                "CCBlade _solve_nonlinear_brent failed")

    def _first_bracket(self, inputs, outputs, residuals):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        turbine = self.options['turbine']

        # parameters
        npts = 10  # number of discretization points to find bracket in residual solve

        swapsign = 1 if turbine else -1
        Vx = inputs['Vx'] * swapsign
        theta = inputs['theta'] * swapsign
        Vy = inputs['Vy']

        # quadrants
        epsilon = 1e-6
        q1 = [epsilon, np.pi/2]
        q2 = [-np.pi/2, -epsilon]
        q3 = [np.pi/2, np.pi-epsilon]
        q4 = [-np.pi+epsilon, -np.pi/2]

        # ---- determine quadrants based on case -----
        Vx_is_zero = np.isclose(Vx, 0.0, atol=1e-6)
        Vy_is_zero = np.isclose(Vy, 0.0, atol=1e-6)

        # I'll just be lame for now and use loops.
        phi_1 = np.zeros((num_nodes, num_radial))
        phi_2 = np.zeros((num_nodes, num_radial))
        success = np.tile(False, (num_nodes, num_radial))
        for i in range(num_nodes):
            for j in range(num_radial):

                if Vx_is_zero[i, j] and Vy_is_zero[i, j]:
                    success[i, j] = False
                    continue

                elif Vx_is_zero[i, j]:

                    startfrom90 = False  # start bracket search from 90 deg instead of 0 deg.

                    if Vy[i, j] > 0 and theta[i, j] > 0:
                        order = (q1, q2)
                    elif Vy[i, j] > 0 and theta[i, j] < 0:
                        order = (q2, q1)
                    elif Vy[i, j] < 0 and theta[i, j] > 0:
                        order = (q3, q4)
                    else:  # Vy < 0 and theta < 0
                        order = (q4, q3)

                elif Vy_is_zero[i, j]:

                    startfrom90 = True  # start bracket search from 90 deg

                    if Vx[i, j] > 0 and abs(theta[i, j]) < np.pi/2:
                        order = (q1, q3)
                    elif Vx[i, j] < 0 and abs(theta[i, j]) < np.pi/2:
                        order = (q2, q4)
                    elif Vx[i, j] > 0 and abs(theta[i, j]) > np.pi/2:
                        order = (q3, q1)
                    else:  # Vx[i, j] < 0 and abs(theta[i, j]) > np.pi/2
                        order = (q4, q2)

                else:  # normal case

                    startfrom90 = False

                    if Vx[i, j] > 0 and Vy[i, j] > 0:
                        order = (q1, q2, q3, q4)
                    elif Vx[i, j] < 0 and Vy[i, j] > 0:
                        order = (q2, q1, q4, q3)
                    elif Vx[i, j] > 0 and Vy[i, j] < 0:
                        order = (q3, q4, q1, q2)
                    else:  # Vx < 0 and Vy < 0
                        order = (q4, q3, q2, q1)

                for (phimin, phimax) in order:  # quadrant orders.  In most cases it should find root in first quadrant searched.

                    # check to see if it would be faster to reverse the bracket search direction
                    backwardsearch = False
                    if not startfrom90:
                        # if phimin == -pi/2 || phimax == -pi/2:  # q2 or q4
                        if np.isclose(phimin, -np.pi/2) or np.isclose(phimax, -np.pi/2):  # q2 or q4
                            backwardsearch = True
                    else:
                        if np.isclose(phimax, np.pi/2):  # q1
                            backwardsearch = True

                    # find bracket
                    found, p1, p2 = self._first_bracket_search(
                        inputs, outputs, residuals, i, j,
                        phimin, phimax, npts, backwardsearch)
                    success[i, j], phi_1[i, j], phi_2[i, j] = found, p1, p2

                    # once bracket is found, return it.
                    if success[i, j]:
                        break

        return success, phi_1, phi_2

    def _first_bracket_search(self, inputs, outputs, residuals,
                              i, j,
                              xmin, xmax, n, backwardsearch):

        xvec = np.linspace(xmin, xmax, n)
        if backwardsearch:  # start from xmax and work backwards
            xvec = xvec[::-1]

        # fprev = f(xvec[1])
        outputs['phi'][i, j] = xvec[0]
        self.apply_nonlinear(inputs, outputs, residuals)
        fprev = residuals['phi'][i, j]
        for k in range(1, n):
            # fnext = f(xvec[i])
            outputs['phi'][i, j] = xvec[k]
            self.apply_nonlinear(inputs, outputs, residuals)
            fnext = residuals['phi'][i, j]
            if fprev*fnext < 0:  # bracket found
                if backwardsearch:
                    return True, xvec[k], xvec[k-1]
                else:
                    return True, xvec[k-1], xvec[k]
            fprev = fnext

        return False, 0.0, 0.0


class FunctionalsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('num_blades', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_input('radii', shape=(num_nodes, num_radial), units='m')
        self.add_input('dradii', shape=(num_nodes, num_radial), units='m')
        self.add_input('Np',
                       shape=(num_nodes, num_radial), units='N/m')
        self.add_input('Tp',
                       shape=(num_nodes, num_radial), units='N/m')
        self.add_input('omega', shape=num_nodes, units='rad/s')
        self.add_input('v', shape=num_nodes, units='m/s')

        self.add_output('thrust', shape=num_nodes, units='N')
        self.add_output('torque', shape=num_nodes, units='N*m')
        self.add_output('power', shape=num_nodes, units='W')
        self.add_output('efficiency', shape=num_nodes, val=10.)

        self.declare_partials('thrust', 'dradii')
        self.declare_partials('torque', 'dradii')
        self.declare_partials('torque', 'radii')
        self.declare_partials('thrust', 'Np')
        self.declare_partials('torque', 'Tp')

        self.declare_partials('power', 'dradii')
        self.declare_partials('power', 'radii')
        self.declare_partials('power', 'Np')
        self.declare_partials('power', 'Tp')

        self.declare_partials('efficiency', 'dradii')
        self.declare_partials('efficiency', 'radii')
        self.declare_partials('efficiency', 'Tp')
        self.declare_partials('efficiency', 'Np')
        self.declare_partials('efficiency', 'omega')
        self.declare_partials('efficiency', 'v')

        # turn on dynamic partial coloring
        self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs):
        B = self.options['num_blades']
        radii = inputs['radii']
        dradii = inputs['dradii']
        Np = inputs['Np']
        Tp = inputs['Tp']
        v = inputs['v'][:, np.newaxis]
        omega = inputs['omega'][:, np.newaxis]

        thrust = outputs['thrust'][:] = B*np.sum(Np * dradii, axis=1)
        torque = outputs['torque'][:] = B*np.sum(Tp * radii * dradii, axis=1)
        outputs['power'][:] = outputs['torque']*inputs['omega']

        outputs['efficiency'] = (thrust*v[:, 0])/(torque*omega[:, 0])


class CCBladeGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('num_blades', types=int)
        self.options.declare('airfoil_interp')
        self.options.declare('turbine', types=bool)
        self.options.declare(
            'phi_residual_solve_nonlinear', types=str, default='brent',
            check_valid=lambda _, s: s in ['bracketing', 'brent'])

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        num_blades = self.options['num_blades']
        airfoil_interp = self.options['airfoil_interp']
        turbine = self.options['turbine']
        solve_nonlinear = self.options['phi_residual_solve_nonlinear']

        comp = om.ExecComp(
            ['hub_radius = 0.5*hub_diameter',
             'prop_radius = 0.5*prop_diameter'],
            shape=num_nodes, has_diag_partials=True,
            hub_radius={'units': 'm'},
            hub_diameter={'units': 'm'},
            prop_radius={'units': 'm'},
            prop_diameter={'units': 'm'})
        self.add_subsystem('hub_prop_radius_comp', comp, promotes=['*'])

        comp = LocalInflowAngleComp(
            num_nodes=num_nodes, num_radial=num_radial, num_blades=num_blades,
            turbine=turbine,
            airfoil_interp=airfoil_interp, debug_print=False,
            solve_nonlinear=solve_nonlinear)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        self.add_subsystem('ccblade_comp', comp,
                           promotes_inputs=[
                               'radii', 'chord', 'theta', 'Vx', 'Vy',
                               'rho', 'mu', 'asound', 'hub_radius',
                               'prop_radius', 'precone'],
                           promotes_outputs=['Np', 'Tp'])

        comp = FunctionalsComp(num_nodes=num_nodes, num_radial=num_radial,
                               num_blades=num_blades)
        self.add_subsystem(
            'ccblade_torquethrust_comp', comp,
            promotes_inputs=['radii', 'dradii', 'Np', 'Tp', 'v', 'omega'],
            promotes_outputs=['thrust', 'torque', 'power', 'efficiency'])

        self.linear_solver = om.DirectSolver(assemble_jac=True)
