import unittest

import numpy as np 

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
from ccblade import ccblade_py as ccb


def dummy_airfoil(alpha, Re, Mach):
    cl = 2*np.pi*alpha
    cd = np.zeros_like(cl)

    return cl, cd


class SeparateCompTestCase(unittest.TestCase): 

    def test_local_inflow_implicit_solve_turbine(self): 

        num_nodes = 1
        # B = 3  # number of blades
        precone = 0.
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

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
                                        num_radial=num_radial,
                                        airfoil_interp=dummy_airfoil,
                                        turbine=turbine,
                                        debug_print=False)

        prob.model.add_subsystem('ccblade', comp)

        prob.setup()

        # prob['ccblade.phi'] = np.array([0.706307, 0.416522, 0.226373, 0.153094,
        #                                 0.115256, 0.092305])[np.newaxis, :]
        prob['ccblade.phi'] = 1. # initial guess
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

        expected_phi = np.array([0.66632224, 0.39417967, 0.20869982, 0.13808477, 0.10209614,
                   0.08053812])

        assert_rel_error(self, expected_phi, prob['ccblade.phi'][0], 1e-5)
        # prob.model.list_outputs(residuals=True, print_arrays=True)


    def test_local_inflow_implicit_solve_propeller(self): 

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

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
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

        expected_phi = np.array([-0.63890233, -0.6121064 , -0.58740212, -0.56450533, -0.54329884,
                                 -0.52111391, -0.48818073, -0.45717514, -0.42966735, -0.40510076,
                                 -0.38302422, -0.36308034, -0.34496445, -0.32845977, -0.31336418,
                                 -0.29949603, -0.28672351, -0.27492358, -0.26399077, -0.25384599,
                                 -0.24441014, -0.23561878, -0.22742504, -0.21978496, -0.21267805,
                                 -0.20608672, -0.20001685, -0.19453287, -0.18977539, -0.18584404,
                                 -0.18203572])

        assert_rel_error(self, expected_phi, prob['ccblade.phi'][0], 1e-5)


if __name__ == "__main__": 

    unittest.main()