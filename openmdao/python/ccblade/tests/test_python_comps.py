import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
from ccblade.inflow import SimpleInflow, WindTurbineInflow
from ccblade import ccblade_py as ccb
from ccblade.utils import af_from_files


class CCBladePythonTestCase(unittest.TestCase):

    def test_turbine_grant_ingram(self):
        # Copied from CCBlade.jl/test/runtests.jl

        # --- verification using example from "Wind Turbine Blade Analysis using the Blade Element Momentum Method"
        # --- by Grant Ingram: https://community.dur.ac.uk/g.l.ingram/download/wind_turbine_design.pdf

        # Note: There were various problems with the pdf data. Fortunately the excel
        # spreadsheet was provided: http://community.dur.ac.uk/g.l.ingram/download.php

        # - They didn't actually use the NACA0012 data.  According to the spreadsheet they just used cl = 0.084*alpha with alpha in degrees.
        # - There is an error in the spreadsheet where CL at the root is always 0.7.  This is because the classical approach doesn't converge properly at that station.
        # - the values for gamma and chord were rounded in the pdf.  The spreadsheet has more precise values.
        # - the tip is not actually converged (see significant error for Delta A).  I converged it further using their method.

        num_nodes = 1

        # --- rotor definition ---
        turbine = True
        hub_radius = 0.01
        prop_radius = 5.0
        prop_radius_eff = 500.0
        B = 3  # number of blades
        precone = 0.

        r = np.array([0.2, 1, 2, 3, 4, 5])[np.newaxis, :]
        gamma = np.array([61.0, 74.31002131, 84.89805553, 89.07195504, 91.25038415, 92.58003871])
        theta = ((90.0 - gamma)*np.pi/180)[np.newaxis, :]
        chord = np.array([0.7, 0.706025153, 0.436187551, 0.304517933, 0.232257636, 0.187279622])[np.newaxis, :]

        num_radial = r.size

        def affunc(alpha, Re, M):
            cl = 0.084*180/np.pi*alpha
            return cl, np.zeros_like(cl)

        # --- inflow definitions ---
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
        prob['ccblade.pitch'] = 0.0

        prob.run_model()

        # First entry in phi is uncomparable because the classical method fails at the root so they fixed cl
        expected_beta = np.array([66.1354, 77.0298, 81.2283, 83.3961, 84.7113])
        beta = 90.0 - np.degrees(prob['ccblade.phi'][0, 1:])
        assert_rel_error(self, beta, expected_beta, 1e-5)

        expected_a = np.array([0.2443, 0.2497, 0.2533, 0.2556, 0.25725])
        assert_rel_error(self, prob['ccblade.a'][0, 1:], expected_a, 1e-3)

        expected_ap = np.array([0.0676, 0.0180, 0.0081, 0.0046, 0.0030])
        assert_rel_error(self, prob['ccblade.ap'][0, 1:], expected_ap, 5e-3)

    def test_propeller_example(self):
        # Example from the tutorial in the CCBlade documentation.

        num_nodes = 1
        turbine = False

        Rhub = np.array([0.5 * 0.0254]).reshape((1, 1))
        Rtip = np.array([3.0 * 0.0254]).reshape((1, 1))
        num_blades = 2
        precone = np.array([0.0]).reshape((1, 1))

        r = 0.0254*np.array(
            [0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332, 1.1128,
             1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911, 1.6708, 1.7505,
             1.8302, 1.9099, 1.9896, 2.0693, 2.1490, 2.2287, 2.3084, 2.3881,
             2.4678, 2.5475, 2.6273, 2.7070, 2.7867, 2.8661, 2.9410]).reshape(
                 1, -1)
        num_radial = r.shape[-1]

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

        rho = np.array([1.225]).reshape((num_nodes, 1))
        Vinf = np.array([10.0]).reshape((num_nodes, 1))
        omega = np.array([8000.0*(2*np.pi/60.0)]).reshape((num_nodes, 1))

        Vy = omega*r*np.cos(precone)
        Vx = np.tile(Vinf * np.cos(precone), (1, num_radial))
        mu = np.array([1.]).reshape((num_nodes, 1))
        asound = np.array([1.]).reshape((num_nodes, 1))
        pitch = np.array([0.0]).reshape((num_nodes, 1))

        # Airfoil interpolator.
        af = af_from_files(["airfoils/NACA64_A17.dat"])[0]

        prob = om.Problem()

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
                                        num_radial=num_radial,
                                        num_blades=num_blades,
                                        airfoil_interp=af,
                                        turbine=turbine,
                                        debug_print=False)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.add_subsystem('ccblade', comp)

        prob.setup()
        prob.final_setup()

        prob['ccblade.phi'] = 1.
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
        prob['ccblade.pitch'] = pitch

        prob.run_model()

        Nptest = np.array(
            [1.8660880922356378, 2.113489633244873, 2.35855792055661,
             2.60301402945597, 2.844874233881403, 3.1180230827072126,
             3.560077224628854, 4.024057801497014, 4.480574891998562,
             4.9279550384928275, 5.366395080074933, 5.79550918136406,
             6.21594163851808, 6.622960527391846, 7.017012349498324,
             7.3936834781240774, 7.751945902955048, 8.086176029603802,
             8.393537672577372, 8.67090062789216, 8.912426896510306,
             9.111379449026037, 9.264491105426602, 9.361598738055728,
             9.397710628068818, 9.360730779314666, 9.236967116872792,
             9.002418776792911, 8.617229305924996, 7.854554211296309,
             5.839491141636506])
        Tptest = np.array(
            [1.481919153409856, 1.5816880353415623, 1.6702432911163534,
             1.7502397903925069, 1.822089134395204, 1.8965254874252537,
             2.0022647148294554, 2.097706171361262, 2.178824887386094,
             2.2475057498944886, 2.3058616094094666, 2.355253913018444,
             2.3970643308370168, 2.4307239254050717, 2.4574034513165794,
             2.4763893383410522, 2.488405728268889, 2.492461784055084,
             2.4887264544021237, 2.4772963155708783, 2.457435891854637,
             2.4282986089025607, 2.3902927838322237, 2.3418848562229155,
             2.283388513786012, 2.2134191689454954, 2.130781255778788,
             2.0328865955896, 1.9153448642630952, 1.7308451522888118,
             1.2736544011110416])
        unormtest = np.array(
            [0.1138639166930624, 0.1215785884908478, 0.12836706426605704,
             0.13442694075150818, 0.13980334658952898, 0.14527249541450538,
             0.15287706861957473, 0.15952130275430465, 0.16497198426904225,
             0.16942902209255595, 0.17308482185019125, 0.17607059901193356,
             0.17850357997819633, 0.1803781237713692, 0.18177831882512596,
             0.18267665167815783, 0.18311924527883217, 0.18305367052760668,
             0.182487470134022, 0.1814205780851561, 0.17980424363698078,
             0.1775794222894007, 0.1747493664535781, 0.1712044765873724,
             0.1669243629724566, 0.16177907188793106, 0.155616714947619,
             0.14815634397029886, 0.13888287431637295, 0.12464247057085576,
             0.09292645450646951])
        vnormtest = np.array(
            [0.09042291182918308, 0.090986677078917, 0.09090479653774426,
             0.09038728871285233, 0.08954144817337718, 0.08836143378908702,
             0.08598138211326231, 0.08315706129440237, 0.08022297890584436,
             0.07727195122065926, 0.07437202068064472, 0.07155384528141737,
             0.06883664444997291, 0.0662014244622726, 0.06365995181515205,
             0.061184457505937574, 0.05878201223442723, 0.056423985398129595,
             0.05410845965501752, 0.05183227774671869, 0.04957767474023305,
             0.04732717658478895, 0.04508635659098153, 0.04282827989707323,
             0.04055808783300249, 0.03825394697198818, 0.0358976247398962,
             0.033456013675480394, 0.030869388594900515, 0.027466462150913407,
             0.020268236545135546])

        assert_array_almost_equal(prob.get_val('ccblade.Np', units='N/m')[0, :], Nptest, decimal=2)
        assert_array_almost_equal(prob.get_val('ccblade.Tp', units='N/m')[0, :], Tptest, decimal=2)
        assert_array_almost_equal((prob.get_val('ccblade.u', units='m/s')/Vinf)[0, :], unormtest, decimal=2)
        assert_array_almost_equal((prob.get_val('ccblade.v', units='m/s')/Vinf)[0, :], vnormtest, decimal=2)

    def test_turbine_example(self):
        # Example from the tutorial in the CCBlade documentation.

        num_nodes = 1
        turbine = True

        Rhub = np.array([1.5]).reshape((num_nodes, 1))
        Rtip = np.array([63.0]).reshape((num_nodes, 1))
        num_blades = 3
        precone = np.array([2.5*np.pi/180.0]).reshape((num_nodes, 1))

        r = np.array(
            [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500,
             24.0500, 28.1500, 32.2500, 36.3500, 40.4500, 44.5500,
             48.6500, 52.7500, 56.1667, 58.9000, 61.6333]).reshape(num_nodes, -1)
        num_radial = r.shape[-1]

        chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249,
                          4.007, 3.748, 3.502, 3.256, 3.010, 2.764, 2.518,
                          2.313, 2.086, 1.419]).reshape((num_nodes, num_radial))

        theta = np.pi/180*np.array([13.308, 13.308, 13.308, 13.308, 11.480,
                                    10.162, 9.011, 7.795, 6.544, 5.361, 4.188,
                                    3.125, 2.319, 1.526, 0.863, 0.370,
                                    0.106]).reshape((num_nodes, num_radial))

        rho = np.array([1.225]).reshape((num_nodes, 1))
        vhub = np.array([10.0]).reshape((num_nodes, 1))
        tsr = 7.55
        rotorR = Rtip*np.cos(precone)
        omega = vhub*tsr/rotorR

        yaw = np.array([0.0]).reshape((num_nodes, 1))
        tilt = np.array([5.0*np.pi/180.0]).reshape((num_nodes, 1))
        hub_height = np.array([90.0]).reshape((num_nodes, 1))
        shear_exp = np.array([0.2]).reshape((num_nodes, 1))
        azimuth = np.array([0.0]).reshape((num_nodes, 1))
        pitch = np.array([0.0]).reshape((num_nodes, 1))

        mu = np.array([1.]).reshape((num_nodes, 1))
        asound = np.array([1.]).reshape((num_nodes, 1))

        # Define airfoils. In this case we have 8 different airfoils that we
        # load into an array. These airfoils are defined in files.
        airfoil_fnames = ["airfoils/Cylinder1.dat", "airfoils/Cylinder2.dat",
                          "airfoils/DU40_A17.dat", "airfoils/DU35_A17.dat",
                          "airfoils/DU30_A17.dat", "airfoils/DU25_A17.dat",
                          "airfoils/DU21_A17.dat", "airfoils/NACA64_A17.dat"]
        aftypes = af_from_files(airfoil_fnames)

        # indices correspond to which airfoil is used at which station
        af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

        # create airfoil array, adjusting for Python's zero-based indexing.
        airfoils = [aftypes[i-1] for i in af_idx]

        prob = om.Problem()

        comp = om.IndepVarComp()
        comp.add_output('vhub', val=vhub, units='m/s')
        comp.add_output('omega', val=omega, units='rad/s')
        comp.add_output('radii', val=r, units='m')
        comp.add_output('chord', val=chord, units='m')
        comp.add_output('theta', val=theta, units='rad')
        comp.add_output('rho', val=rho, units='kg/m**3')
        comp.add_output('mu', val=mu, units='N/m**2*s')
        comp.add_output('asound', val=asound, units='m/s')
        comp.add_output('precone', val=precone, units='rad')
        comp.add_output('yaw', val=yaw, units='rad')
        comp.add_output('tilt', val=tilt, units='rad')
        comp.add_output('hub_height', val=hub_height, units='m')
        comp.add_output('shear_exp', val=shear_exp)
        comp.add_output('azimuth', val=azimuth, units='rad')
        comp.add_output('hub_radius', val=Rhub, units='m')
        comp.add_output('prop_radius', val=Rtip, units='m')
        comp.add_output('pitch', val=pitch, units='rad')
        prob.model.add_subsystem('ivc', comp, promotes=['*'])

        comp = WindTurbineInflow(num_nodes=num_nodes,
                                 num_radial=num_radial)
        prob.model.add_subsystem('inflow', comp, promotes=['*'])

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
                                        num_radial=num_radial,
                                        num_blades=num_blades,
                                        airfoil_interp=airfoils,
                                        turbine=turbine,
                                        debug_print=False)

        prob.model.add_subsystem('ccblade', comp, promotes=['*'])

        prob.setup()
        prob.final_setup()

        prob.run_model()

        expected_phi = np.array([1.24151, 0.984853, 0.794433, 0.474428,
                                 0.360879, 0.306092, 0.261674, 0.221899,
                                 0.19487, 0.170476, 0.151866, 0.142013,
                                 0.129982, 0.118756, 0.108178, 0.0976137,
                                 0.0887128])
        assert_rel_error(self, expected_phi, prob['ccblade.phi'][0, :], 1e-5)

    def test_propeller_aero4students(self):
        # Copied from CCBlade.jl/test/runtests.jl

        # -------- verification: propellers.  using script at http://www.aerodynamics4students.com/propulsion/blade-element-propeller-theory.php ------
        # I increased their tolerance to 1e-6

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
        theta = np.arctan(pitch/(2*np.pi*r))

        def affunc(alpha, Re, M):
            cl = 6.2*alpha
            cd = 0.008 - 0.003*cl + 0.01*cl*cl
            return cl, cd

        prob = om.Problem()
        num_radial = r.shape[-1]

        comp = om.IndepVarComp()
        comp.add_output('v', val=np.arange(1, 60+1)[:num_nodes], units='m/s')
        comp.add_output('omega', val=np.tile(RPM, num_nodes), units='rpm')
        comp.add_output('radii', val=r, units='m')
        comp.add_output('chord', val=chord, shape=(num_nodes, num_radial), units='m')
        comp.add_output('theta', val=theta, shape=(num_nodes, num_radial), units='rad')
        comp.add_output('precone', val=0., shape=num_nodes, units='rad')
        comp.add_output('rho', val=rho, shape=(num_nodes, 1), units='kg/m**3')
        comp.add_output('mu', val=1.0, shape=(num_nodes, 1), units='N/m**2*s')
        comp.add_output('asound', val=1.0, shape=(num_nodes, 1), units='m/s')
        comp.add_output('hub_radius', val=Rhub_eff, shape=num_nodes, units='m')
        comp.add_output('prop_radius', val=Rtip_eff, shape=num_nodes, units='m')
        comp.add_output('pitch', val=0.0, shape=num_nodes, units='rad')
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
                             'precone', 'pitch'],
            promotes_outputs=['Np', 'Tp'])

        prob.setup()
        prob.final_setup()
        prob.run_model()

        Np = prob.get_val('Np', units='N/m')
        Tp = prob.get_val('Tp', units='N/m')
        T = np.sum(Np*dr, axis=1)*B
        Q = np.sum(r*Tp*dr, axis=1)*B

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

        assert_rel_error(self, T, tsim[:num_nodes], 1e-5)
        assert_rel_error(self, Q, qsim[:num_nodes], 1e-5)

        # Run with v = 20.0 m/s.
        prob.set_val('v', 20.0, units='m/s', indices=0)
        prob.run_model()

        Vhub = prob.get_val('v', units='m/s', indices=0)
        omega = prob.get_val('omega', units='rad/s', indices=0)
        Np = prob.get_val('Np', units='N/m', indices=0)
        Tp = prob.get_val('Tp', units='N/m', indices=0)
        T = np.sum(Np*dr)*B
        Q = np.sum(r[0, :]*Tp*dr)*B

        P = Q * omega
        n = omega/(2*np.pi)
        CT = T/(rho*n**2*D**4)
        CQ = Q/(rho*n**2*D**5)
        eff = T*Vhub/P

        assert_rel_error(self, CT, 0.056110238632657, 1e-6)
        assert_rel_error(self, CQ, 0.004337202960642, 1e-6)
        assert_rel_error(self, eff, 0.735350632777002, 1e-5)

    def test_camber(self):
        # Copied from CCBlade.jl/test/runtests.jl

        # inputs
        chord = 0.10
        D = 1.6
        RPM = 2100
        rho = 1.225
        pitch = 1.0  # pitch distance in meters.

        # --- rotor definition ---
        turbine = False
        Rhub = 0.0
        Rtip = D/2
        Rhub_eff = 1e-6  # something small to eliminate hub effects
        Rtip_eff = 100.0  # something large to eliminate tip effects
        B = 2  # number of blades

        # --- section definitions ---

        num_nodes = 1
        R = D/2.0
        r = np.linspace(R/10, R, 11)
        dr = r[1] - r[0]
        r = np.tile(r, (num_nodes, 1))
        dr = np.tile(dr, r.shape)
        theta = np.arctan(pitch/(2*np.pi*r))

        def affunc(alpha, Re, M):
            alpha0 = -3*np.pi/180
            cl = 6.2*(alpha - alpha0)
            cd = 0.008 - 0.003*cl + 0.01*cl*cl
            return cl, cd

        prob = om.Problem()
        num_radial = r.shape[-1]

        comp = om.IndepVarComp()
        comp.add_output('v', val=[5.0], units='m/s')
        comp.add_output('omega', val=np.tile(RPM, num_nodes), units='rpm')
        comp.add_output('radii', val=r, units='m')
        comp.add_output('dradii', val=dr, units='m')
        comp.add_output('chord', val=chord, shape=(num_nodes, num_radial), units='m')
        comp.add_output('theta', val=theta, shape=(num_nodes, num_radial), units='rad')
        comp.add_output('precone', val=0., shape=num_nodes, units='rad')
        comp.add_output('rho', val=rho, shape=(num_nodes, 1), units='kg/m**3')
        comp.add_output('mu', val=1.0, shape=(num_nodes, 1), units='N/m**2*s')
        comp.add_output('asound', val=1.0, shape=(num_nodes, 1), units='m/s')
        comp.add_output('hub_radius_eff', val=Rhub_eff, shape=num_nodes, units='m')
        comp.add_output('prop_radius_eff', val=Rtip_eff, shape=num_nodes, units='m')
        comp.add_output('hub_radius', val=Rhub, shape=num_nodes, units='m')
        comp.add_output('prop_radius', val=Rtip, shape=num_nodes, units='m')
        comp.add_output('pitch', val=0.0, shape=num_nodes, units='rad')
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
                             'mu', 'asound', ('hub_radius', 'hub_radius_eff'),
                             ('prop_radius', 'prop_radius_eff'), 'precone',
                             'pitch'],
            promotes_outputs=['Np', 'Tp'])

        prob.setup()
        prob.final_setup()
        prob.run_model()

        Np = prob.get_val('Np', units='N/m')
        Tp = prob.get_val('Tp', units='N/m')
        T = np.sum(Np*dr)*B
        Q = np.sum(r*Tp*dr)*B

        assert_rel_error(self, 1223.0506862888788, T, 1e-9)
        assert_rel_error(self, 113.79919472569034, Q, 1e-10)

        assert_rel_error(self, prob.get_val('Np', units='N/m')[0, 3], 427.3902632382494, 1e-10)
        assert_rel_error(self, prob.get_val('Tp', units='N/m')[0, 3], 122.38414345762305, 1e-10)
        assert_rel_error(self, prob.get_val('ccblade.a')[0, 3], 2.2845512476210943, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.ap')[0, 3], 0.05024950801920044, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.u', units='m/s')[0, 3], 11.422756238105471, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.v', units='m/s')[0, 3], 3.2709314141649575, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.phi', units='rad')[0, 3], 0.2596455971546484, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.alpha', units='rad')[0, 3], 0.23369406105568025, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.W', units='m/s')[0, 3], 63.96697566502531, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.cl')[0, 3], 1.773534419416163, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.cd')[0, 3], 0.03413364011028978, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.cn')[0, 3], 1.7053239640124302, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.ct')[0, 3], 0.48832327407767123, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.F')[0, 3], 1.0, 1e-8)
        assert_rel_error(self, prob.get_val('ccblade.G')[0, 3], 1.0, 1e-8)

        theta = np.arctan(pitch/(2*np.pi*r)) - 3*np.pi/180
        prob.set_val('theta', theta, units='rad')
        prob.run_model()

        Np = prob.get_val('Np', units='N/m')
        Tp = prob.get_val('Tp', units='N/m')
        T = np.sum(Np*dr)*B
        Q = np.sum(r*Tp*dr)*B

        assert_rel_error(self, T, 1e3*0.962407923825140, 1e-6)
        assert_rel_error(self, Q, 1e2*0.813015017103876, 1e-6)

    def test_multiple_adv_ratios(self):
        # num_nodes = 20
        nstart = 0
        nend = 19
        num_nodes = nend - nstart + 1

        r = 0.0254*np.array(
            [0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332, 1.1128,
             1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911, 1.6708, 1.7505,
             1.8302, 1.9099, 1.9896, 2.0693, 2.1490, 2.2287, 2.3084, 2.3881,
             2.4678, 2.5475, 2.6273, 2.7070, 2.7867, 2.8661, 2.9410]).reshape(
                 1, -1)
        num_radial = r.shape[-1]
        r = np.tile(r, (num_nodes, 1))

        chord = 0.0254*np.array(
            [0.6270, 0.6255, 0.6231, 0.6199, 0.6165, 0.6125, 0.6054, 0.5973,
             0.5887, 0.5794, 0.5695, 0.5590, 0.5479, 0.5362, 0.5240, 0.5111,
             0.4977, 0.4836, 0.4689, 0.4537, 0.4379, 0.4214, 0.4044, 0.3867,
             0.3685, 0.3497, 0.3303, 0.3103, 0.2897, 0.2618, 0.1920]).reshape(
                 (1, num_radial))
        chord = np.tile(chord, (num_nodes, 1))

        theta = np.pi/180.0*np.array(
            [40.2273, 38.7657, 37.3913, 36.0981, 34.8803, 33.5899, 31.6400,
             29.7730, 28.0952, 26.5833, 25.2155, 23.9736, 22.8421, 21.8075,
             20.8586, 19.9855, 19.1800, 18.4347, 17.7434, 17.1005, 16.5013,
             15.9417, 15.4179, 14.9266, 14.4650, 14.0306, 13.6210, 13.2343,
             12.8685, 12.5233, 12.2138]).reshape((1, num_radial))
        theta = np.tile(theta, (num_nodes, 1))

        rho = 1.225
        Rhub = 0.0254*.5
        Rtip = 0.0254*3.0
        B = 2  # number of blades
        turbine = False

        J = np.linspace(0.1, 0.9, 20)[nstart:nend+1]  # advance ratio
        # J = np.linspace(0.1, 0.9, 20)

        omega = 8000.0*np.pi/30

        n = omega/(2*np.pi)
        D = 2*Rtip

        Vinf = J*D*n
        dr = r[0, 1] - r[0, 0]

        # Airfoil interpolator.
        af = af_from_files(["airfoils/NACA64_A17.dat"])[0]

        prob = om.Problem()

        comp = om.IndepVarComp()
        comp.add_output('v', val=Vinf, units='m/s')
        comp.add_output('omega', val=np.tile(omega, num_nodes), units='rad/s')
        comp.add_output('radii', val=r, shape=(num_nodes, num_radial), units='m')
        comp.add_output('dradii', val=dr, shape=(num_nodes, num_radial), units='m')
        comp.add_output('chord', val=chord, shape=(num_nodes, num_radial), units='m')
        comp.add_output('theta', val=theta, shape=(num_nodes, num_radial), units='rad')
        comp.add_output('precone', val=0., shape=num_nodes, units='rad')
        comp.add_output('rho', val=rho, shape=(num_nodes, 1), units='kg/m**3')
        comp.add_output('mu', val=1.0, shape=(num_nodes, 1), units='N/m**2*s')
        comp.add_output('asound', val=1.0, shape=(num_nodes, 1), units='m/s')
        comp.add_output('hub_radius', val=Rhub, shape=num_nodes, units='m')
        comp.add_output('prop_radius', val=Rtip, shape=num_nodes, units='m')
        comp.add_output('pitch', val=0.0, shape=num_nodes, units='rad')
        prob.model.add_subsystem('ivc', comp, promotes_outputs=['*'])

        comp = SimpleInflow(num_nodes=num_nodes, num_radial=num_radial)
        prob.model.add_subsystem(
            "simple_inflow", comp,
            promotes_inputs=["v", "omega", "radii", "precone"],
            promotes_outputs=["Vx", "Vy"])

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
                                        num_radial=num_radial,
                                        num_blades=B,
                                        airfoil_interp=af,
                                        turbine=turbine,
                                        debug_print=False)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.model.add_subsystem(
            'ccblade', comp,
            promotes_inputs=['radii', 'chord', 'theta', 'Vx', 'Vy', 'rho',
                             'mu', 'asound', 'hub_radius', 'prop_radius',
                             'precone', 'pitch'],
            promotes_outputs=['Np', 'Tp'])

        comp = ccb.FunctionalsComp(num_nodes=num_nodes, num_radial=num_radial,
                                   num_blades=B)
        prob.model.add_subsystem(
            'ccblade_torquethrust_comp', comp,
            promotes_inputs=['hub_radius', 'prop_radius', 'radii', 'Np', 'Tp', 'v', 'omega'],
            promotes_outputs=['thrust', 'torque', 'efficiency'])

        prob.setup()
        prob.final_setup()
        prob.run_model()

        etatest = np.array(
            [0.24598190455626265, 0.3349080300487075, 0.4155652767326253,
             0.48818637673414306, 0.5521115225679999, 0.6089123481436948,
             0.6595727776885079, 0.7046724703349897, 0.7441662053086512,
             0.7788447616541276, 0.8090611349633181, 0.8347808848055981,
             0.8558196582739432, 0.8715046719672315, 0.8791362131978436,
             0.8670633642311274, 0.7974063895510229, 0.2715632768892098, 0.0,
             0.0])[nstart:nend+1]

        thrust = prob.get_val('thrust', units='N')
        eff = prob.get_val('efficiency')
        assert_array_almost_equal(eff[thrust > 0.0], etatest[thrust > 0.0], decimal=2)

    def test_hover(self):
        # num_nodes = 40
        nstart = 0
        nend = 39
        num_nodes = nend - nstart + 1
        chord = 0.060
        theta = 0.0
        Rtip = 0.656
        Rhub = 0.19*Rtip
        rho = 1.225
        omega = 800.0*np.pi/30
        Vinf = 0.0
        turbine = False
        B = 3

        r = np.linspace(Rhub + 0.01*Rtip, Rtip - 0.01*Rtip, 30)
        num_radial = r.shape[-1]
        r = np.tile(r, (num_nodes, 1))

        chord = np.tile(chord, (num_nodes, num_radial))
        theta = np.tile(theta, (num_nodes, num_radial))

        # Airfoil interpolator.
        af = af_from_files(["airfoils/naca0012v2.txt"])[0]

        pitch = np.linspace(1e-4, 20*np.pi/180, 40)[nstart:nend+1]
        prob = om.Problem()

        comp = om.IndepVarComp()
        comp.add_output('v', val=Vinf, shape=num_nodes, units='m/s')
        comp.add_output('omega', val=np.tile(omega, num_nodes), units='rad/s')
        comp.add_output('radii', val=r, shape=(num_nodes, num_radial), units='m')
        comp.add_output('chord', val=chord, shape=(num_nodes, num_radial), units='m')
        comp.add_output('theta', val=theta, shape=(num_nodes, num_radial), units='rad')
        comp.add_output('precone', val=0., shape=num_nodes, units='rad')
        comp.add_output('rho', val=rho, shape=(num_nodes, 1), units='kg/m**3')
        comp.add_output('mu', val=1.0, shape=(num_nodes, 1), units='N/m**2*s')
        comp.add_output('asound', val=1.0, shape=(num_nodes, 1), units='m/s')
        comp.add_output('hub_radius', val=Rhub, shape=num_nodes, units='m')
        comp.add_output('prop_radius', val=Rtip, shape=num_nodes, units='m')
        comp.add_output('pitch', val=pitch, shape=num_nodes, units='rad')
        prob.model.add_subsystem('ivc', comp, promotes_outputs=['*'])

        comp = SimpleInflow(num_nodes=num_nodes, num_radial=num_radial)
        prob.model.add_subsystem(
            "simple_inflow", comp,
            promotes_inputs=["v", "omega", "radii", "precone"],
            promotes_outputs=["Vx", "Vy"])

        comp = ccb.LocalInflowAngleComp(num_nodes=num_nodes,
                                        num_radial=num_radial,
                                        num_blades=B,
                                        airfoil_interp=af,
                                        turbine=turbine,
                                        debug_print=False)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.model.add_subsystem(
            'ccblade', comp,
            promotes_inputs=['radii', 'chord', 'theta', 'Vx', 'Vy', 'rho',
                             'mu', 'asound', 'hub_radius', 'prop_radius',
                             'precone', 'pitch'],
            promotes_outputs=['Np', 'Tp'])

        comp = ccb.FunctionalsComp(num_nodes=num_nodes, num_radial=num_radial,
                                   num_blades=B)
        prob.model.add_subsystem(
            'ccblade_torquethrust_comp', comp,
            promotes_inputs=['hub_radius', 'prop_radius', 'radii', 'Np', 'Tp', 'v', 'omega'],
            promotes_outputs=['thrust', 'torque', 'efficiency'])

        prob.setup()
        prob.final_setup()
        prob.run_model()

        # these are not directly from the experimental data, but have been compared to the experimental data and compare favorably.
        # this is more of a regression test on the Vx=0 case
        CTcomp = np.array([9.452864991304056e-9, 6.569947366946672e-5,
                           0.00022338783939012262, 0.0004420355541809959,
                           0.0007048495858030926, 0.0010022162314665929,
                           0.0013268531109981317, 0.0016736995380106938,
                           0.0020399354072946035, 0.0024223576277264307,
                           0.0028189858460418893, 0.0032281290309981213,
                           0.003649357660426685, 0.004081628946214875,
                           0.004526034348853718, 0.004982651929181267,
                           0.0054553705714941, 0.005942700094508395,
                           0.006447634897014323, 0.006963626871239654,
                           0.007492654931894796, 0.00803866268066438,
                           0.008597914974368199, 0.009163315934297088,
                           0.00973817187875574, 0.010309276997090536,
                           0.010827599471613264, 0.011322361524464346,
                           0.01180210507896255, 0.012276543435307877,
                           0.012749323136224754, 0.013223371028562213,
                           0.013697833731701945, 0.01417556699620018,
                           0.014646124777465859, 0.015112116772851365,
                           0.015576452747370885, 0.01602507607909594,
                           0.016461827164870473, 0.016880126012974343])
        CQcomp = np.array([0.000226663607327854, 0.0002270862930229147,
                           0.0002292742856722754, 0.00023412703235791698,
                           0.00024192624628054639, 0.0002525855612031453,
                           0.00026638347417704255, 0.00028314784456601373,
                           0.00030299181501156373, 0.0003259970210015136,
                           0.00035194661281707764, 0.00038102864688744595,
                           0.0004132249034847219, 0.00044859355432807347,
                           0.0004873204055790553, 0.0005293656187218555,
                           0.0005753409000182888, 0.0006250099998058788,
                           0.0006788861946930185, 0.0007361096750412038,
                           0.0007970800153713466, 0.0008624036743669367,
                           0.0009315051772818803, 0.0010035766105979213,
                           0.0010791941808362153, 0.0011566643573792704,
                           0.001229236439467123, 0.0013007334425769355,
                           0.001372124993921022, 0.0014449961686871802,
                           0.0015197156782734364, 0.0015967388663224156,
                           0.0016761210460920718, 0.0017578748614666766,
                           0.0018409716992061841, 0.0019248522013432586,
                           0.0020103360819251357, 0.002096387027559033,
                           0.002182833604491109, 0.0022686470790128036])
        T = prob.get_val('thrust', units='N')
        Q = prob.get_val('torque', units='N*m')
        A = np.pi*Rtip**2
        CT = T/(rho * A * (omega*Rtip)**2)
        CQ = Q/(rho * A * (omega*Rtip)**2 * Rtip)

        assert_array_almost_equal(CT, CTcomp[nstart:nend+1], decimal=3)
        assert_array_almost_equal(CQ, CQcomp[nstart:nend+1], decimal=3)


if __name__ == "__main__":

    unittest.main()
