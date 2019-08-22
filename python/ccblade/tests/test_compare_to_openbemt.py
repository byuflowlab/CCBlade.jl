import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
from openbemt.airfoils.process_airfoils import ViternaAirfoil
from openbemt.bemt.groups.bemt_group import BEMTGroup
from ccblade.geometry import GeometryGroup
from ccblade.inflow import SimpleInflow
from ccblade.ccblade_py import CCBladeGroup


class OpenBEMTTestCase(unittest.TestCase):

    def test_openbemt_analysis(self):
        interp = ViternaAirfoil().create_akima(
            'mh117', Re_scaling=False, extend_alpha=True)

        def ccblade_interp(alpha, Re, Mach):
            shape = alpha.shape
            x = np.concatenate(
                [
                    alpha.flatten()[:, np.newaxis],
                    Re.flatten()[:, np.newaxis]
                ], axis=-1)
            y = interp(x)
            y.shape = shape + (2,)
            return y[..., 0], y[..., 1]

        num_nodes = 1
        num_blades = 3
        num_radial = 15
        num_cp = 6
        chord = 10.
        theta = np.linspace(65., 25., num_cp)*np.pi/180.
        pitch = 0.
        prop_data = {
            'num_radial': num_radial,
            'num_cp': num_cp,
            'pitch': pitch,
            'chord': chord,
            'theta': theta,
            'spline_type': 'akima',
            'B': num_blades,
            'interp': interp}

        hub_diameter = 30.  # cm
        prop_diameter = 150.  # cm
        c0 = np.sqrt(1.4*287.058*300.)  # meters/second
        rho0 = 1.4*98600./(c0*c0)  # kg/m^3
        omega = 236.

        prob = om.Problem()

        comp = om.IndepVarComp()
        comp.add_discrete_input('B', val=num_blades)
        comp.add_output('rho', val=rho0, shape=num_nodes, units='kg/m**3')
        comp.add_output('mu', val=1., shape=num_nodes, units='N/m**2*s')
        comp.add_output('asound', val=c0, shape=num_nodes, units='m/s')
        comp.add_output('v', val=77.2, shape=num_nodes, units='m/s')
        comp.add_output('alpha', val=0., shape=num_nodes, units='rad')
        comp.add_output('incidence', val=0., shape=num_nodes, units='rad')
        comp.add_output('precone', val=0., units='deg')
        comp.add_output('omega', val=omega, shape=num_nodes, units='rad/s')
        comp.add_output('hub_diameter', val=hub_diameter, shape=num_nodes,
                        units='cm')
        comp.add_output('prop_diameter', val=prop_diameter, shape=num_nodes,
                        units='cm')
        comp.add_output('pitch', val=pitch, shape=num_nodes, units='rad')
        comp.add_output('chord_dv', val=chord, shape=num_cp, units='cm')
        comp.add_output('theta_dv', val=theta, shape=num_cp, units='rad')
        prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

        prob.model.add_subsystem(
            'bemt_group', BEMTGroup(num_nodes=num_nodes, prop_data=prop_data),
            promotes_inputs=['rho', 'mu', 'v', 'alpha', 'incidence', 'omega',
                             'hub_diameter', 'prop_diameter', 'pitch',
                             'chord_dv', 'theta_dv'],
            promotes_outputs=[('normal_load_dist', 'openbemt_normal_load'),
                              ('circum_load_dist', 'openbemt_circum_load')])

        comp = GeometryGroup(num_nodes=num_nodes, num_cp=num_cp,
                             num_radial=num_radial)
        prob.model.add_subsystem(
            'geometry_group', comp,
            promotes_inputs=['hub_diameter', 'prop_diameter', 'chord_dv',
                             'theta_dv', 'pitch'],
            promotes_outputs=['radii', 'dradii', 'chord', 'theta'])

        comp = SimpleInflow(num_nodes=num_nodes, num_radial=num_radial)
        prob.model.add_subsystem(
            'inflow_comp', comp,
            promotes_inputs=['v', 'omega', 'radii', 'precone'],
            promotes_outputs=['Vx', 'Vy'])

        comp = CCBladeGroup(num_nodes=num_nodes, num_radial=num_radial,
                            airfoil_interp=ccblade_interp,
                            turbine=False)
        prob.model.add_subsystem(
            'ccblade_group', comp,
            promotes_inputs=['B', 'radii', 'dradii', 'chord', 'theta', 'rho',
                             'mu', 'asound', 'Vx', 'Vy', 'v', 'precone',
                             'omega', 'hub_diameter', 'prop_diameter'],
            promotes_outputs=[('Np', 'ccblade_normal_load'),
                              ('Tp', 'ccblade_circum_load')])

        prob.setup()
        prob.final_setup()
        prob.run_model()

        # Get the normal and circumferential loads.
        num_blades = prob.get_val('inputs_comp.B')
        dradii = prob.get_val('dradii', units='m')
        ccblade_normal_load = prob.get_val(
            'ccblade_normal_load', units='N/m')*num_blades
        ccblade_circum_load = prob.get_val(
            'ccblade_circum_load', units='N/m')*num_blades

        openbemt_normal_load = prob.get_val('openbemt_normal_load', units='N')
        openbemt_circum_load = prob.get_val('openbemt_circum_load', units='N')
        openbemt_normal_load /= dradii
        openbemt_circum_load /= dradii

        assert_rel_error(self, openbemt_normal_load, ccblade_normal_load, 1e-8)
        assert_rel_error(self, openbemt_circum_load, ccblade_circum_load, 1e-8)

    def test_openbemt_optimization(self):

        interp = ViternaAirfoil().create_akima(
            'mh117', Re_scaling=False, extend_alpha=True)

        def ccblade_interp(alpha, Re, Mach):
            shape = alpha.shape
            x = np.concatenate(
                [
                    alpha.flatten()[:, np.newaxis],
                    Re.flatten()[:, np.newaxis]
                ], axis=-1)
            y = interp(x)
            y.shape = shape + (2,)
            return y[..., 0], y[..., 1]

        num_nodes = 1
        num_blades = 3
        num_radial = 15
        num_cp = 6
        chord = 10.
        theta = np.linspace(65., 25., num_cp)*np.pi/180.
        pitch = 0.
        prop_data = {
            'num_radial': num_radial,
            'num_cp': num_cp,
            'pitch': pitch,
            'chord': chord,
            'theta': theta,
            'spline_type': 'akima',
            'B': num_blades,
            'interp': interp}

        hub_diameter = 30.  # cm
        prop_diameter = 150.  # cm
        c0 = np.sqrt(1.4*287.058*300.)  # meters/second
        rho0 = 1.4*98600./(c0*c0)  # kg/m^3
        omega = 236.

        prob_ccblade = om.Problem()
        prob_openbemt = om.Problem()

        comp = om.IndepVarComp()
        comp.add_discrete_input('B', val=num_blades)
        comp.add_output('rho', val=rho0, shape=num_nodes, units='kg/m**3')
        comp.add_output('mu', val=1., shape=num_nodes, units='N/m**2*s')
        comp.add_output('asound', val=c0, shape=num_nodes, units='m/s')
        comp.add_output('v', val=77.2, shape=num_nodes, units='m/s')
        comp.add_output('alpha', val=0., shape=num_nodes, units='rad')
        comp.add_output('incidence', val=0., shape=num_nodes, units='rad')
        comp.add_output('precone', val=0., units='deg')
        comp.add_output('omega', val=omega, shape=num_nodes, units='rad/s')
        comp.add_output('hub_diameter', val=hub_diameter, shape=num_nodes,
                        units='cm')
        comp.add_output('prop_diameter', val=prop_diameter, shape=num_nodes,
                        units='cm')
        comp.add_output('pitch', val=pitch, shape=num_nodes, units='rad')
        comp.add_output('chord_dv', val=chord, shape=num_cp, units='cm')
        comp.add_output('theta_dv', val=theta, shape=num_cp, units='rad')
        prob_ccblade.model.add_subsystem('indep_var_comp', comp,
                                         promotes=['*'])
        prob_openbemt.model.add_subsystem('indep_var_comp', comp,
                                          promotes=['*'])

        comp = GeometryGroup(num_nodes=num_nodes, num_cp=num_cp,
                             num_radial=num_radial)
        prob_ccblade.model.add_subsystem(
            'geometry_group', comp,
            promotes_inputs=['hub_diameter', 'prop_diameter', 'chord_dv',
                             'theta_dv', 'pitch'],
            promotes_outputs=['radii', 'dradii', 'chord', 'theta'])

        comp = SimpleInflow(num_nodes=num_nodes, num_radial=num_radial)
        prob_ccblade.model.add_subsystem(
            'inflow_comp', comp,
            promotes_inputs=['v', 'omega', 'radii', 'precone'],
            promotes_outputs=['Vx', 'Vy'])

        comp = CCBladeGroup(num_nodes=num_nodes, num_radial=num_radial,
                            airfoil_interp=ccblade_interp, turbine=False,
                            phi_residual_solve_nonlinear=True)
        prob_ccblade.model.add_subsystem(
            'ccblade_group', comp,
            promotes_inputs=['B', 'radii', 'dradii', 'chord', 'theta', 'rho',
                             'mu', 'asound', 'v', 'omega', 'Vx', 'Vy',
                             'precone', 'hub_diameter', 'prop_diameter'],
            promotes_outputs=['thrust', 'torque', 'efficiency'])

        prob_ccblade.model.add_design_var('chord_dv', lower=1., upper=20.,
                                          scaler=5e-2)
        prob_ccblade.model.add_design_var('theta_dv', lower=20.*np.pi/180.,
                                          upper=90*np.pi/180.)

        prob_ccblade.model.add_objective('efficiency', scaler=-1.,)
        prob_ccblade.model.add_constraint('thrust', equals=700., scaler=1e-3,
                                          indices=np.arange(num_nodes))
        prob_ccblade.driver = om.pyOptSparseDriver()
        prob_ccblade.driver.options['optimizer'] = 'SNOPT'

        prob_ccblade.setup()
        prob_ccblade.final_setup()
        prob_ccblade.run_driver()

        num_blades = prob_ccblade.get_val('indep_var_comp.B')
        ccblade_normal_load = prob_ccblade.get_val(
            'ccblade_group.Np', units='N/m')*num_blades
        ccblade_circum_load = prob_ccblade.get_val(
            'ccblade_group.Tp', units='N/m')*num_blades

        prob_openbemt.model.add_subsystem(
            'bemt_group', BEMTGroup(num_nodes=num_nodes, prop_data=prop_data),
            promotes_inputs=['rho', 'mu', 'v', 'alpha', 'incidence', 'omega',
                             'hub_diameter', 'prop_diameter', 'pitch',
                             'chord_dv', 'theta_dv'],
            promotes_outputs=[('normal_load_dist', 'openbemt_normal_load'),
                              ('circum_load_dist', 'openbemt_circum_load'),
                              'dradii', 'thrust', 'efficiency'])

        prob_openbemt.model.add_design_var('chord_dv', lower=1., upper=20.,
                                           scaler=5e-2)
        prob_openbemt.model.add_design_var('theta_dv', lower=20.*np.pi/180.,
                                           upper=90*np.pi/180.)

        prob_openbemt.model.add_objective('efficiency', scaler=-1.,)
        prob_openbemt.model.add_constraint('thrust', equals=700., scaler=1e-3,
                                           indices=np.arange(num_nodes))
        prob_openbemt.driver = om.pyOptSparseDriver()
        prob_openbemt.driver.options['optimizer'] = 'SNOPT'

        prob_openbemt.setup()
        prob_openbemt.final_setup()
        prob_openbemt.model.bemt_group.phi_group.nonlinear_solver.options['iprint'] = 0
        prob_openbemt.run_driver()

        openbemt_normal_load = prob_openbemt.get_val(
            'openbemt_normal_load', units='N')
        openbemt_circum_load = prob_openbemt.get_val(
            'openbemt_circum_load', units='N')
        dradii = prob_openbemt.get_val('dradii', units='m')
        openbemt_normal_load /= dradii
        openbemt_circum_load /= dradii

        assert_rel_error(self, ccblade_normal_load, openbemt_normal_load, 5e-5)
        assert_rel_error(self, ccblade_circum_load, openbemt_circum_load, 5e-5)


if __name__ == "__main__":

    unittest.main()
