import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
from ccblade.inflow import SimpleInflow
from ccblade import ccblade_py as ccb


def dummy_airfoil(alpha, Re, Mach):
    cl = 2*np.pi*alpha
    cd = np.zeros_like(cl)

    return cl, cd


class SeparateCompTestCase(unittest.TestCase):

    def test_local_inflow_implicit_solve_turbine(self):

        num_nodes = 1
        B = 3  # number of blades
        precone = 0.
        turbine = True
        hub_radius = 0.01
        prop_radius = 5.0
        prop_radius_eff = 500.0
        r = np.array([0.2, 1, 2, 3, 4, 5])[np.newaxis, :]
        num_radial = r.size

        gamma = np.array([61.0, 74.31002131, 84.89805553, 89.07195504,
                          91.25038415, 92.58003871])
        theta = ((90.0 - gamma)*np.pi/180)[np.newaxis, :]
        chord = np.array([0.7, 0.706025153, 0.436187551, 0.304517933,
                          0.232257636, 0.187279622])[np.newaxis, :]

        def affunc(alpha, Re, M):
            cl = 0.084*alpha*180/np.pi

            return cl, np.zeros_like(cl)

        Vinf = np.array([7.0]).reshape((num_nodes, 1))
        tsr = 8
        omega = tsr*Vinf/prop_radius
        rho = 1.0

        Vx = np.tile(Vinf * np.cos(precone), (1, num_radial))
        Vy = omega*r*np.cos(precone)

        prob = om.Problem()

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
                                        num_radial=num_radial,
                                        num_blades=B,
                                        airfoil_interp=affunc,
                                        turbine=turbine,
                                        debug_print=False)

        prob.model.add_subsystem('ccblade', comp)

        prob.setup()

        prob['ccblade.phi'] = 1.  # initial guess
        prob['ccblade.radii'] = r
        prob['ccblade.chord'] = chord
        prob['ccblade.theta'] = theta
        prob['ccblade.Vx'] = Vx
        prob['ccblade.Vy'] = Vy
        prob['ccblade.rho'] = rho
        prob['ccblade.mu'] = 1.
        prob['ccblade.asound'] = 1.
        prob['ccblade.hub_radius'] = hub_radius
        prob['ccblade.prop_radius'] = prop_radius_eff
        prob['ccblade.precone'] = precone

        prob.run_model()

        # First entry in phi is uncomparable because the classical method fails at the root so they fixed cl
        expected_beta = np.array([66.1354, 77.0298, 81.2283, 83.3961, 84.7113])
        beta = 90.0 - np.degrees(prob['ccblade.phi'][0, 1:])
        assert_rel_error(self, beta, expected_beta, 1e-5)

        expected_a = np.array([0.2443, 0.2497, 0.2533, 0.2556, 0.25725])
        assert_rel_error(self, prob['ccblade.a'][0, 1:], expected_a, 1e-3)

        expected_ap = np.array([0.0676, 0.0180, 0.0081, 0.0046, 0.0030])
        assert_rel_error(self, prob['ccblade.ap'][0, 1:], expected_ap, 5e-3)

    def test_local_inflow_implicit_solve_propeller(self):

        num_nodes = 1
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
                                        num_blades=3,
                                        airfoil_interp=af,
                                        turbine=turbine,
                                        debug_print=False)

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

    def test_propeller_aero4students(self):
        # From test/runtests.jl

        # inputs
        chord = 0.10
        D = 1.6
        RPM = 2100
        rho = 1.225
        pitch = 1.0  # pitch distance in meters.

        # --- rotor definition ---
        turbine = False
        # Rhub = 0.0
        # Rtip = D/2
        Rhub_eff = 1e-6  # something small to eliminate hub effects
        Rtip_eff = 100.0  # something large to eliminate tip effects
        B = 2  # number of blades

        # --- section definitions ---

        num_nodes = 60
        R = D/2.0
        r = np.linspace(R/10, R, 11)
        dr = r[1] - r[0]
        r = np.tile(r, (num_nodes, 1))
        # print(f"r =\n{r}")
        dr = np.tile(dr, r.shape)
        theta = np.arctan(pitch/(2*np.pi*r))

        def affunc(alpha, Re, M):
            cl = 6.2*alpha
            cd = 0.008 - 0.003*cl + 0.01*cl*cl
            return cl, cd

        prob = om.Problem()
        num_radial = r.shape[-1]

        comp = om.IndepVarComp()
        comp.add_output('v', val=np.arange(1, num_nodes+1), units='m/s')
        comp.add_output('omega', val=np.tile(RPM, num_nodes), units='rpm')
        comp.add_output('radii', val=r, units='m')
        comp.add_output('dradii', val=dr, units='m')
        comp.add_output('chord', val=chord, shape=(num_nodes, num_radial), units='m')
        comp.add_output('theta', val=theta, shape=(num_nodes, num_radial), units='rad')
        comp.add_output('precone', val=0., shape=num_nodes, units='rad')
        comp.add_output('rho', val=rho, shape=(num_nodes, 1), units='kg/m**3')
        comp.add_output('mu', val=1.0, shape=(num_nodes, 1), units='N/m**2*s')
        comp.add_output('asound', val=1.0, shape=(num_nodes, 1), units='m/s')
        comp.add_output('hub_radius', val=Rhub_eff, shape=num_nodes, units='m')
        comp.add_output('prop_radius', val=Rtip_eff, shape=num_nodes, units='m')
        prob.model.add_subsystem('ivc', comp, promotes_outputs=['*'])

        comp = SimpleInflow(num_nodes=num_nodes, num_radial=num_radial)
        prob.model.add_subsystem(
            "simple_inflow", comp,
            promotes_inputs=["v", "omega", "radii", "precone"],
            promotes_outputs=["Vx", "Vy"])

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
                                        num_radial=num_radial,
                                        num_blades=B,
                                        airfoil_interp=affunc,
                                        turbine=turbine,
                                        debug_print=False)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.model.add_subsystem(
            'ccblade', comp,
            promotes_inputs=['radii', 'chord', 'theta', 'Vx', 'Vy', 'rho',
                             'mu', 'asound', 'hub_radius', 'prop_radius',
                             'precone'],
            promotes_outputs=['Np', 'Tp'])

        comp = ccb.FunctionalsComp(num_nodes=num_nodes, num_radial=num_radial,
                                   num_blades=B)
        prob.model.add_subsystem(
            'ccblade_torquethrust_comp', comp,
            promotes_inputs=['radii', 'dradii', 'Np', 'Tp', 'v', 'omega'],
            promotes_outputs=['thrust', 'torque'])

        prob.setup()
        prob.final_setup()
        prob.run_model()

        tsim = 1e3*np.array(
            [1.045361193032356, 1.025630300048415, 1.005234466788998,
             0.984163367036026, 0.962407923825140, 0.939960208707079,
             0.916813564966455, 0.892962691000145, 0.868403981825492,
             0.843134981103815, 0.817154838249790, 0.790463442573673,
             0.763063053839278, 0.734956576558370, 0.706148261507327,
             0.676643975451150, 0.646450304160057, 0.615575090105131,
             0.584027074365864, 0.551815917391907, 0.518952127358381,
             0.485446691671386, 0.451311288662196, 0.416557935286392,
             0.381199277009438, 0.345247916141561, 0.308716772800348,
             0.271618894441869, 0.233967425339051, 0.195775319296371,
             0.157055230270717, 0.117820154495231, 0.078082266879117,
             0.037854059197644, -0.002852754149850, -0.044026182837742,
             -0.085655305814570, -0.127728999394140, -0.170237722799272,
             -0.213169213043848, -0.256515079286031, -0.300266519551194,
             -0.344414094748869, -0.388949215983616, -0.433863576642539,
             -0.479150401337354, -0.524801553114807, -0.570810405128802,
             -0.617169893200684, -0.663873474163182, -0.710915862524620,
             -0.758291877949762, -0.805995685105502, -0.854022273120508,
             -0.902366919041604, -0.951025170820984, -0.999992624287163,
             -1.049265666456123, -1.098840222937414, -1.148712509929845])
        qsim = 1e2*np.array(
            [0.803638686218187, 0.806984572453978, 0.809709290183008,
             0.811743686838315, 0.813015017103876, 0.813446921530685,
             0.812959654049620, 0.811470393912576, 0.808893852696513,
             0.805141916379142, 0.800124489784850, 0.793748780791057,
             0.785921727832179, 0.776548246109426, 0.765532528164390,
             0.752778882688809, 0.738190986274448, 0.721673076180745,
             0.703129918771009, 0.682467282681955, 0.659592296506578,
             0.634413303042323, 0.606840565246423, 0.576786093366321,
             0.544164450503912, 0.508891967461804, 0.470887571011192,
             0.430072787279711, 0.386371788290446, 0.339711042057184,
             0.290019539402947, 0.237229503458026, 0.181274942660876,
             0.122093307308376, 0.059623821454727, -0.006190834182631,
             -0.075406684829235, -0.148076528546541, -0.224253047813501,
             -0.303980950928302, -0.387309291734422, -0.474283793689904,
             -0.564946107631716, -0.659336973911858, -0.757495165410553,
             -0.859460291551374, -0.965266648683888, -1.074949504731187,
             -1.188540970723477, -1.306072104649531, -1.427575034895290,
             -1.553080300508925, -1.682614871422754, -1.816205997296014,
             -1.953879956474228, -2.095662107769925, -2.241576439746701,
             -2.391647474158875, -2.545897099743367, -2.704346566395035])

        assert_rel_error(self, tsim, prob['thrust'], 1e-5)
        assert_rel_error(self, qsim, prob['torque'], 1e-5)


if __name__ == "__main__":

    unittest.main()
