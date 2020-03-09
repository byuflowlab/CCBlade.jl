import warnings
import numpy as np

import openmdao.api as om
from ccblade.utils import get_rows_cols
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
        self.add_input('pitch', shape=(num_nodes, 1), units='rad')

        self.add_output('phi', shape=(num_nodes, num_radial), units='rad')
        self.add_output('Np', shape=(num_nodes, num_radial), units='N/m')
        self.add_output('Tp', shape=(num_nodes, num_radial), units='N/m')
        self.add_output('a', shape=(num_nodes, num_radial))
        self.add_output('ap', shape=(num_nodes, num_radial))
        self.add_output('u', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('v', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('alpha', shape=(num_nodes, num_radial), units='rad')
        self.add_output('W', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('cl', shape=(num_nodes, num_radial))
        self.add_output('cd', shape=(num_nodes, num_radial))
        self.add_output('cn', shape=(num_nodes, num_radial))
        self.add_output('ct', shape=(num_nodes, num_radial))
        self.add_output('F', shape=(num_nodes, num_radial))
        self.add_output('G', shape=(num_nodes, num_radial))

        self.declare_partials('*', '*')

        deriv_method = 'fd'
        self.declare_coloring(wrt='*', method=deriv_method, perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    # @profile
    def apply_nonlinear(self, inputs, outputs, residuals):

        #warnings.simplefilter("error")

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
        pitch = inputs['pitch']

        phi = outputs['phi']

        # check if turbine or propeller and change input sign if necessary
        swapsign = 1 if turbine else -1
        # theta = swapsign * inputs['theta']
        # Vx = swapsign * inputs['Vx']
        theta = inputs['theta']
        Vx = inputs['Vx']

        # constants
        sigma_p = B*chord/(2.0*np.pi*r)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        # angle of attack
        alpha = phi - (theta + pitch)

        # Reynolds number
        W0 = np.sqrt(Vx*Vx + Vy*Vy)  # ignoring induction, which is generally a very minor error and only affects Reynolds number
        Re = rho * W0 * chord / mu

        # Mach number
        Mach = W0/asound  # also ignoring induction

        # airfoil cl/cd
        cl = np.zeros(alpha.shape, dtype=alpha.dtype)
        cd = cl.copy()
        if turbine:
            for i in range(num_radial):
                cl[:, i], cd[:, i] = af[i](alpha[:, i], Re[:, i], Mach[:, i])
        else:
            for i in range(num_radial):
                # I don't know why the airfoil interpolation function sometimes returns 1D arrays and sometimes 2D.
                CL, CD = af[i](-alpha[:, i], Re[:, i], Mach[:, i])
                cl[:, i] = np.squeeze(CL)
                cd[:, i] = np.squeeze(CD)
            cl *= -1

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

        # parameters used in Vx=0 and Vy=0 cases
        k0 = cn*sigma_p/(4.0*F*sphi*cphi)
        k0p = ct*sigma_p/(4.0*F*sphi*sphi)

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

        u = np.zeros(phi.shape, dtype=phi.dtype)
        v = u.copy()
        a = u.copy()
        ap = u.copy()
        G = u.copy()

        vyfilt = np.zeros(Vx.shape, dtype=bool)

        # Vy approx 0 region
        vyfilt[np.abs(Vy) < 1e-6] = True
        vxflt = Vx[vyfilt]
        v[vyfilt] = k0p[vyfilt]*abs_cs(vxflt)
        fflt = F[vyfilt]
        residuals['phi'][vyfilt] = np.sign(vxflt)*4.*fflt*sphi[vyfilt]*cphi[vyfilt] - ct[vyfilt]*sigma_p[vyfilt]
        G[vyfilt] = fflt
        del vxflt
        del fflt

        # Vx approx 0 region
        vxfilt = np.logical_not(vyfilt)
        vxfilt = np.logical_and(vxfilt, Vx < 1e-6)

        sgnphiflt = np.sign(phi[vxfilt])
        u[vxfilt] = sgnphiflt * k0[vxfilt] * Vy[vxfilt]
        fflt = F[vxfilt]
        residuals['phi'][vxfilt] = sphi[vxfilt]**2 + 0.25*sgnphiflt*cn[vxfilt]*sigma_p[vxfilt]/fflt
        G[vxfilt] = np.sqrt(fflt)
        del sgnphiflt
        del fflt

        filt = np.logical_not(np.logical_or(vyfilt, vxfilt))
        del vxfilt

        # Vx != 0 and  Vy != 0
        phi_neg = phi < 0.0
        k[phi_neg] *= -1.
        del phi_neg

        momfilt = np.logical_and(filt, k <= 2./3.)
        kfilt = k[momfilt]
        a[momfilt] = kfilt/(1. + kfilt)
        del kfilt

        empfilt = np.logical_and(filt, k > 2./3.)
        empF = F[empfilt]
        empk = k[empfilt]

        g2 = np.ones(F.shape, dtype=F.dtype)
        g3 = np.ones(F.shape, dtype=F.dtype)
        g2[empfilt] = 2.0*empF*empk - (4./3.-empF)*empF
        g3[empfilt] = 2.0*empF*empk - (25./9.-2.*empF)

        zfilt = np.logical_and(empfilt, np.abs(g3) < 1e-6)
        a[zfilt] = 1.0 - .5*np.sqrt(g2[zfilt])
        del zfilt

        nzfilt = np.logical_and(empfilt, np.abs(g3) >= 1e-6)
        fflt = F[nzfilt]
        g1 = 2.0*fflt*k[nzfilt] - (10.0/9-fflt)
        a[nzfilt] = (g1 - np.sqrt(g2[nzfilt])) / g3[nzfilt]
        del fflt

        vxflt = Vx[filt]
        vyflt = Vy[filt]
        aflt = a[filt]
        one_m_aflt = 1.0 - aflt
        u[filt] = aflt * vxflt

        # -------- tangential induction ----------

        kp[filt] *= np.sign(vxflt)

        ap[filt] = kp[filt]/(1 - kp[filt])
        v[filt] = ap[filt] * vyflt

        residuals['phi'][filt] = sphi[filt]/one_m_aflt - vxflt/vyflt*cphi[filt]/(1. + ap[filt])
        G[filt] = (1.0 - np.sqrt(1.0 - 4*aflt * one_m_aflt*F[filt]))/(2.*aflt)

        del aflt
        del one_m_aflt
        del vyflt
        del vxflt

        ####################################
        # Slow Pythonized Code
        ####################################
        # for i in range(num_nodes):
        #     for j in range(num_radial):
                # if np.isclose(Vx[i, j], 0.0, atol=1e-6):
                #     u[i, j] = np.sign(phi[i, j])*k0[i, j]*Vy[i, j]
                #     # v[i, j] = 0.0  # already 0
                #     # a[i, j] = 0.0
                #     # ap[i, j] = 0.0
                #     residuals['phi'][i, j] = np.sin(phi[i, j])**2 + np.sign(phi[i, j])*cn[i, j]*sigma_p[i, j]/(4.0*F[i, j])
                #     G[i, j] = np.sqrt(F[i, j])
                # elif np.isclose(Vy[i, j], 0.0, atol=1e-6):
                #     # u[i, j] = 0.0
                #     v[i, j] = k0p[i, j]*abs_cs(Vx[i, j])
                #     # a[i, j] = 0.0
                #     # ap[i, j] = 0.0
                #     residuals['phi'][i, j] = np.sign(Vx[i, j])*4*F[i, j]*sphi[i, j]*cphi[i, j] - ct[i, j]*sigma_p[i, j]
                #     G[i, j] = F[i, j]
                # else:
                #     if phi[i, j] < 0.0:
                #         k[i, j] *= -1.0

                #     if k[i, j] <= 2./3.:  # momentum region
                #         a[i, j] = k[i, j]/(1 + k[i, j])

                #     else:  # empirical region
                #         g2 = 2.0*F[i, j]*k[i, j] - (4.0/3-F[i, j])*F[i, j]
                #         g3 = 2.0*F[i, j]*k[i, j] - (25.0/9-2*F[i, j])
                #         if np.isclose(g3, 0.0, atol=1e-6):  # avoid singularity
                #             a[i, j] = 1.0 - 1.0/(2.0*np.sqrt(g2))
                #         else:
                #             g1 = 2.0*F[i, j]*k[i, j] - (10.0/9-F[i, j])
                #             a[i, j] = (g1 - np.sqrt(g2)) / g3

                #     u[i, j] = a[i, j] * Vx[i, j]

                #     # -------- tangential induction ----------
                #     if Vx[i, j] < 0:
                #         kp[i, j] *= -1.0

                #     ap[i, j] = kp[i, j]/(1 - kp[i, j])
                #     v[i, j] = ap[i, j] * Vy[i, j]

                #     residuals['phi'][i, j] = np.sin(phi[i, j])/(1.0 - a[i, j]) - Vx[i, j]/Vy[i, j]*np.cos(phi[i, j])/(1. + ap[i, j])  # 25%
                #     G[i, j] = (1.0 - np.sqrt(1.0 - 4*a[i, j]*(1.0 - a[i, j])*F[i, j]))/(2.*a[i, j])  # 13.5%

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

        # -------- tangential induction ----------
        # kp[Vx < 0.] *= -1.

        # ap = kp/(1 - kp)
        # v = ap * Vy

        # ------- loads ---------
        W = np.sqrt((Vx - u)**2 + (Vy + v)**2)
        tmp = 0.5*rho*W*W*chord
        Np = cn*tmp
        Tp = ct*tmp

        # Tp *= swapsign
        # v *= swapsign

        # The BEM methodology applies hub/tip losses to the loads rather than to the velocities.
        # This is the most common way to implement a BEM, but it means that the raw velocities are misleading
        # as they do not contain any hub/tip loss corrections.
        # To fix this we compute the effective hub/tip losses that would produce the same thrust/torque.
        # In other words:
        # CT = 4 a (1 - a) F = 4 a G (1 - a G)\n
        # This is solved for G, then multiplied against the wake velocities.
        # G = (1.0 - np.sqrt(1.0 - 4*a*(1.0 - a)*F))/(2*a)
        u *= G
        v *= G

        if not turbine:
            Np *= swapsign
            Tp *= swapsign
            a *= swapsign
            ap *= swapsign
            u *= swapsign
            v *= swapsign
            alpha *= swapsign
            cl *= swapsign
            cn *= swapsign
            ct *= swapsign

        # Other residuals.
        residuals['Np'] = Np - outputs['Np']
        residuals['Tp'] = Tp - outputs['Tp']
        residuals['a'] = a - outputs['a']
        residuals['ap'] = ap - outputs['ap']
        residuals['u'] = u - outputs['u']
        residuals['v'] = v - outputs['v']
        residuals['alpha'] = alpha - outputs['alpha']
        residuals['W'] = W - outputs['W']
        residuals['cl'] = cl - outputs['cl']
        residuals['cd'] = cd - outputs['cd']
        residuals['cn'] = cn - outputs['cn']
        residuals['ct'] = ct - outputs['ct']
        residuals['F'] = F - outputs['F']
        residuals['G'] = G - outputs['G']

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
        res_1 = np.zeros(phi_1.shape, dtype=phi.dtype)
        res_2 = res_1.copy()

        outputs['phi'][:, :] = phi_1
        self.apply_nonlinear(inputs, outputs, residuals)
        res_1[:, :] = residuals['phi']

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
        out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'alpha', 'W', 'cl', 'cd', 'cn', 'ct', 'F', 'G')
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
            # print(f"bracket_found =\n{bracket_found}")
            raise om.AnalysisError("CCBlade bracketing failed")

        # Create a wrapper function compatible with the brentv function.
        def f(x):
            outputs['phi'][:, :] = x
            self.apply_nonlinear(inputs, outputs, self._residuals)
            return np.copy(self._residuals['phi'])

        # Find the root.
        phi, steps_taken, success = brentv(f, phi_1, phi_2, tolerance=SOLVE_TOL)

        # Fix up the other outputs.
        out_names = ('Np', 'Tp', 'a', 'ap', 'u', 'v', 'alpha', 'W', 'cl', 'cd', 'cn', 'ct', 'F', 'G')
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

        # parameters
        npts = 20  # number of discretization points to find bracket in residual solve

        Vx = inputs['Vx']
        Vy = inputs['Vy']

        # quadrants
        epsilon = 1e-6
        q1 = [epsilon, np.pi/2, False]
        q2 = [-np.pi/2, -epsilon, True]
        q3 = [np.pi/2, np.pi-epsilon, False]
        q4 = [-np.pi+epsilon, -np.pi/2, True]

        # ---- determine quadrants based on case -----
        # Vx_is_zero = np.isclose(Vx, 0.0, atol=1e-6)
        # Vy_is_zero = np.isclose(Vy, 0.0, atol=1e-6)

        # I'll just be lame for now and use loops.
        phi_1 = np.zeros((num_nodes, num_radial))
        phi_2 = np.zeros((num_nodes, num_radial))
        success = np.tile(False, (num_nodes, num_radial))
        for i in range(num_nodes):
            for j in range(num_radial):

                if Vx[i, j] > 0 and Vy[i, j] > 0:
                    order = (q1, q2, q3, q4)
                elif Vx[i, j] < 0 and Vy[i, j] > 0:
                    order = (q2, q1, q4, q3)
                elif Vx[i, j] > 0 and Vy[i, j] < 0:
                    order = (q3, q4, q1, q2)
                else:  # Vx[i, j] < 0 and Vy[i, j] < 0
                    order = (q4, q3, q2, q1)

                for (phimin, phimax, backwardsearch) in order:  # quadrant orders.  In most cases it should find root in first quadrant searched.

                    # backwardsearch = False
                    # if np.isclose(phimin, -np.pi/2) or np.isclose(phimax, -np.pi/2):  # q2 or q4
                    #     backwardsearch = True

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

        outputs['phi'][i, j] = xvec[0]
        self.apply_nonlinear(inputs, outputs, residuals)
        fprev = residuals['phi'][i, j]
        for k in range(1, n):
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
        self.options.declare('dynamic_coloring', types=bool, default=False)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_input('hub_radius', shape=(num_nodes, 1), units='m')
        self.add_input('prop_radius', shape=(num_nodes, 1), units='m')
        self.add_input('radii', shape=(num_nodes, num_radial), units='m')
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

        if self.options['dynamic_coloring']:
            self.declare_partials('*', '*', method='fd')
            # turn on dynamic partial coloring
            self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5,
                                  num_full_jacs=2, tol=1e-20, orders=20,
                                  show_summary=True, show_sparsity=False)
        else:
            ss_sizes = {'i': num_nodes, 'j': num_radial}

            rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss='i', wrt_ss='ij')

            self.declare_partials('thrust', 'Np', rows=rows, cols=cols)
            self.declare_partials('thrust', 'radii', rows=rows, cols=cols)

            self.declare_partials('torque', 'Tp', rows=rows, cols=cols)
            self.declare_partials('torque', 'radii', rows=rows, cols=cols)

            self.declare_partials('power', 'Tp', rows=rows, cols=cols)
            self.declare_partials('power', 'radii', rows=rows, cols=cols)

            self.declare_partials('efficiency', 'Np', rows=rows, cols=cols)
            self.declare_partials('efficiency', 'Tp', rows=rows, cols=cols)
            self.declare_partials('efficiency', 'radii', rows=rows, cols=cols)

            rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss='i', wrt_ss='i')

            self.declare_partials('thrust', 'hub_radius', rows=rows, cols=cols)
            self.declare_partials('thrust', 'prop_radius', rows=rows, cols=cols)

            self.declare_partials('torque', 'hub_radius', rows=rows, cols=cols)
            self.declare_partials('torque', 'prop_radius', rows=rows, cols=cols)

            self.declare_partials('power', 'hub_radius', rows=rows, cols=cols)
            self.declare_partials('power', 'prop_radius', rows=rows, cols=cols)
            self.declare_partials('power', 'omega', rows=rows, cols=cols)

            self.declare_partials('efficiency', 'hub_radius', rows=rows, cols=cols)
            self.declare_partials('efficiency', 'prop_radius', rows=rows, cols=cols)
            self.declare_partials('efficiency', 'omega', rows=rows, cols=cols)
            self.declare_partials('efficiency', 'v', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        B = self.options['num_blades']

        v = inputs['v']
        omega = inputs['omega']
        dtype = omega.dtype

        radii = np.empty((num_nodes, num_radial+2), dtype=dtype)
        radii[:, 0] = inputs['hub_radius'][:, 0]
        radii[:, 1:-1] = inputs['radii']
        radii[:, -1] = inputs['prop_radius'][:, 0]

        Np = np.empty((num_nodes, num_radial+2), dtype=dtype)
        Np[:, 0] = 0.0
        Np[:, 1:-1] = inputs['Np']
        Np[:, -1] = 0.0
        thrust = outputs['thrust'][:] = B*np.sum((radii[:, 1:] - radii[:, :-1])*0.5*(Np[:, :-1] + Np[:, 1:]), axis=1)

        Tp = np.empty((num_nodes, num_radial+2), dtype=dtype)
        Tp[:, 0] = 0.0
        Tp[:, 1:-1] = inputs['Tp']
        Tp[:, -1] = 0.0
        Tp *= radii
        torque = outputs['torque'][:] = B*np.sum((radii[:, 1:] - radii[:, :-1])*0.5*(Tp[:, :-1] + Tp[:, 1:]), axis=1)

        outputs['power'][:] = torque*omega
        outputs['efficiency'][:] = (thrust*v)/outputs['power']

    def compute_partials(self, inputs, partials):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        B = self.options['num_blades']

        v = inputs['v']
        omega = inputs['omega']
        dtype = omega.dtype

        radii = np.empty((num_nodes, num_radial+2), dtype=dtype)
        radii[:, 0] = inputs['hub_radius'][:, 0]
        radii[:, 1:-1] = inputs['radii']
        radii[:, -1] = inputs['prop_radius'][:, 0]

        Np = np.empty((num_nodes, num_radial+2), dtype=dtype)
        Np[:, 0] = 0.0
        Np[:, 1:-1] = inputs['Np']
        Np[:, -1] = 0.0

        thrust = B*np.sum((radii[:, 1:] - radii[:, :-1])*0.5*(Np[:, :-1] + Np[:, 1:]), axis=1)

        dthrust_dNp = partials['thrust', 'Np']
        dthrust_dNp.shape = (num_nodes, num_radial)
        dthrust_dNp[:, :] = B*(radii[:, 1:-1] - radii[:, :-2])*0.5 + B*(radii[:, 2:] - radii[:, 1:-1])*0.5

        dthrust_dradii = partials['thrust', 'radii']
        dthrust_dradii.shape = (num_nodes, num_radial)
        dthrust_dradii[:, :] = B*0.5*(Np[:, :-2] + Np[:, 1:-1]) - B*0.5*(Np[:, 1:-1] + Np[:, 2:])

        dthrust_dhub_radius = partials['thrust', 'hub_radius']
        dthrust_dhub_radius.shape = (num_nodes,)
        dthrust_dhub_radius[:] = B*(-0.5)*(Np[:, 0] + Np[:, 1])

        dthrust_dprop_radius = partials['thrust', 'prop_radius']
        dthrust_dprop_radius.shape = (num_nodes,)
        dthrust_dprop_radius[:] = B*0.5*(Np[:, -2] + Np[:, -1])

        Tp = np.empty((num_nodes, num_radial+2), dtype=dtype)
        Tp[:, 0] = 0.0
        Tp[:, 1:-1] = inputs['Tp']
        Tp[:, -1] = 0.0

        torque = B*np.sum((radii[:, 1:] - radii[:, :-1])*0.5*(Tp[:, :-1]*radii[:, :-1] + Tp[:, 1:]*radii[:, 1:]), axis=1)

        dtorque_dTp = partials['torque', 'Tp']
        dtorque_dTp.shape = (num_nodes, num_radial)
        dtorque_dTp[:, :] = B*(radii[:, 1:-1] - radii[:, :-2])*0.5*radii[:, 1:-1] + B*(radii[:, 2:] - radii[:, 1:-1])*0.5*radii[:, 1:-1]

        dtorque_dradii = partials['torque', 'radii']
        dtorque_dradii.shape = (num_nodes, num_radial)
        dtorque_dradii[:, :] = B*(0.5)*(Tp[:, :-2]*radii[:, :-2] + Tp[:, 1:-1]*radii[:, 1:-1])
        dtorque_dradii[:, :] += B*(-0.5)*(Tp[:, 1:-1]*radii[:, 1:-1] + Tp[:, 2:]*radii[:, 2:])
        dtorque_dradii[:, :] += B*(radii[:, 2:] - radii[:, 1:-1])*0.5*(Tp[:, 1:-1])
        dtorque_dradii[:, :] += B*(radii[:, 1:-1] - radii[:, :-2])*0.5*(Tp[:, 1:-1])

        dtorque_dhub_radius = partials['torque', 'hub_radius']
        dtorque_dhub_radius.shape = (num_nodes,)
        dtorque_dhub_radius[:] = B*(-0.5)*(Tp[:, 0]*radii[:, 0] + Tp[:, 1]*radii[:, 1])
        dtorque_dhub_radius[:] += B*(radii[:, 1] - radii[:, 0])*0.5*(Tp[:, 0])

        dtorque_dprop_radius = partials['torque', 'prop_radius']
        dtorque_dprop_radius.shape = (num_nodes,)
        dtorque_dprop_radius[:] = B*(0.5)*(Tp[:, -2]*radii[:, -2] + Tp[:, -1]*radii[:, -1])
        dtorque_dprop_radius[:] += B*(radii[:, -1] - radii[:, -2])*0.5*(Tp[:, -1])

        power = torque*omega

        dpower_dTp = partials['power', 'Tp']
        dpower_dTp.shape = (num_nodes, num_radial)
        dpower_dTp[:, :] = dtorque_dTp*omega[:, np.newaxis]

        dpower_dradii = partials['power', 'radii']
        dpower_dradii.shape = (num_nodes, num_radial)
        dpower_dradii[:, :] = dtorque_dradii*omega[:, np.newaxis]

        dpower_dhub_radius = partials['power', 'hub_radius']
        dpower_dhub_radius.shape = (num_nodes,)
        dpower_dhub_radius[:] = dtorque_dhub_radius*omega

        dpower_dprop_radius = partials['power', 'prop_radius']
        dpower_dprop_radius.shape = (num_nodes,)
        dpower_dprop_radius[:] = dtorque_dprop_radius*omega

        dpower_domega = partials['power', 'omega']
        dpower_domega.shape = (num_nodes,)
        dpower_domega[:] = torque

        # efficiency = (thrust*v)/power

        defficiency_dNp = partials['efficiency', 'Np']
        defficiency_dNp.shape = (num_nodes, num_radial)
        defficiency_dNp[:, :] = dthrust_dNp*v[:, np.newaxis]/power[:, np.newaxis]

        defficiency_dTp = partials['efficiency', 'Tp']
        defficiency_dTp.shape = (num_nodes, num_radial)
        defficiency_dTp[:, :] = -(thrust[:, np.newaxis]*v[:, np.newaxis])/(power[:, np.newaxis]*power[:, np.newaxis])*dpower_dTp

        defficiency_dradii = partials['efficiency', 'radii']
        defficiency_dradii.shape = (num_nodes, num_radial)
        defficiency_dradii[:, :] = (power[:, np.newaxis]) * (dthrust_dradii*v[:, np.newaxis])
        defficiency_dradii[:, :] -= (thrust[:, np.newaxis]*v[:, np.newaxis]) * (dpower_dradii)
        defficiency_dradii[:, :] /= power[:, np.newaxis]*power[:, np.newaxis]

        defficiency_dhub_radius = partials['efficiency', 'hub_radius']
        defficiency_dhub_radius.shape = (num_nodes,)
        defficiency_dhub_radius[:] = (power) * (dthrust_dhub_radius*v)
        defficiency_dhub_radius[:] -= (thrust*v) * (dpower_dhub_radius)
        defficiency_dhub_radius[:] /= power*power

        defficiency_dprop_radius = partials['efficiency', 'prop_radius']
        defficiency_dprop_radius.shape = (num_nodes,)
        defficiency_dprop_radius[:] = (power) * (dthrust_dprop_radius*v)
        defficiency_dprop_radius[:] -= (thrust*v) * (dpower_dprop_radius)
        defficiency_dprop_radius[:] /= power*power

        defficiency_domega = partials['efficiency', 'omega']
        defficiency_domega.shape = (num_nodes,)
        defficiency_domega[:] = -(thrust*v)/(power*power)*dpower_domega

        defficiency_dv = partials['efficiency', 'v']
        defficiency_dv.shape = (num_nodes,)
        defficiency_dv[:] = thrust/power

        dthrust_dNp.shape = (-1,)
        dthrust_dradii.shape = (-1,)
        dthrust_dhub_radius.shape = (-1,)
        dthrust_dprop_radius.shape = (-1,)
        dtorque_dTp.shape = (-1,)
        dtorque_dradii.shape = (-1,)
        dtorque_dhub_radius.shape = (-1,)
        dtorque_dprop_radius.shape = (-1,)
        dpower_dTp.shape = (-1,)
        dpower_dradii.shape = (-1,)
        dpower_dhub_radius.shape = (-1,)
        dpower_dprop_radius.shape = (-1,)
        dpower_domega.shape = (-1,)
        defficiency_dNp.shape = (-1,)
        defficiency_dTp.shape = (-1,)
        defficiency_dradii.shape = (-1,)
        defficiency_dhub_radius.shape = (-1,)
        defficiency_dprop_radius.shape = (-1,)
        defficiency_domega.shape = (-1,)
        defficiency_dv.shape = (-1,)


class HubPropRadiusComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        self.add_input('hub_diameter', shape=num_nodes, units='m')
        self.add_input('prop_diameter', shape=num_nodes, units='m')

        self.add_output('hub_radius', shape=num_nodes, units='m')
        self.add_output('prop_radius', shape=num_nodes, units='m')

        self.declare_partials('hub_radius', 'hub_diameter')
        self.declare_partials('prop_radius', 'prop_diameter')

        # turn on dynamic partial coloring
        self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs):
        outputs['hub_radius'][:] = 0.5*inputs['hub_diameter']
        outputs['prop_radius'][:] = 0.5*inputs['prop_diameter']


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

        comp = HubPropRadiusComp(num_nodes=num_nodes)
        self.add_subsystem('hub_prop_radius_comp', comp,
                           promotes_inputs=['hub_diameter', 'prop_diameter'],
                           promotes_outputs=['hub_radius', 'prop_radius'])

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
            promotes_inputs=['hub_radius', 'prop_radius', 'radii', 'Np', 'Tp',
                             'v', 'omega'],
            promotes_outputs=['thrust', 'torque', 'power', 'efficiency'])

        self.linear_solver = om.DirectSolver(assemble_jac=True)


if __name__ == "__main__":
    num_nodes = 2
    num_blades = 3
    num_radial = 4

    radii = np.random.random((num_nodes, num_radial))
    dradii = np.random.random((num_nodes, num_radial))
    Np = np.random.random((num_nodes, num_radial))
    Tp = np.random.random((num_nodes, num_radial))
    omega = np.random.random((num_nodes,))
    v = np.random.random((num_nodes,))

    p = om.Problem()

    comp = om.IndepVarComp()
    comp.add_output('radii', val=radii, units='m')
    comp.add_output('Np', val=Np, units='N/m')
    comp.add_output('Tp', val=Tp, units='N/m')
    comp.add_output('omega', val=omega, units='rad/s')
    comp.add_output('v', val=v, units='m/s')
    p.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = FunctionalsComp(
        num_nodes=num_nodes, num_blades=num_blades, num_radial=num_radial)
    p.model.add_subsystem('functionals_comp', comp, promotes=['*'])

    p.setup()
    p.check_partials()
