import numpy as np
from openmdao.api import ExplicitComponent
import julia.Main as jlmain


class CCBladeWrapperComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('B', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('af_fname', types=str)

        jlmain.eval("""
            using CCBlade
            using PyCall

            function propeller_kernel!(options, inputs, outputs)
                # Airfoil interpolation object.
                af = options["af"]

                # Rotor parameters.
                B = options["B"]
                Rhub = 0.5 * inputs["hub_diameter"][1]
                Rtip = 0.5 * inputs["prop_diameter"][1]
                turbine = false
                rotor = Rotor(Rhub, Rtip, B, turbine)

                # Blade section parameters.
                r = inputs["radii"]
                chord = inputs["chord"]
                theta = inputs["theta"]
                sections = Section.(r, chord, theta, af)

                # Inflow parameters.
                rho = inputs["rho"]
                Vinf = inputs["v"]
                Omega = inputs["omega"]
                inflows = simpleinflow.(Vinf, Omega, r, rho)

                # Solve it!
                solution = solve.(rotor, sections, inflows)

                # Get the distributed loads.
                Np, Tp = loads(solution)

                @. outputs["normal_load_dist"] = Np
                @. outputs["circum_load_dist"] = Tp
            end

            propeller! = pyfunction(
                propeller_kernel!,
                PyDict{String, PyAny}, # options
                PyDict{String, PyArray}, # inputs
                PyDict{String, PyArray}) # outputs
        """)
        self._compute = jlmain.propeller_b

    def setup(self):
        num_radial = self.options['num_radial']
        af_fname = self.options['af_fname']

        self.add_input('rho', units='kg/m**3')
        self.add_input('v', units='m/s')
        self.add_input('omega', units='rad/s')
        self.add_input('hub_diameter', units='m')
        self.add_input('prop_diameter', units='m')
        self.add_input('radii', shape=num_radial, units='m')
        self.add_input('chord', shape=num_radial, units='m')
        self.add_input('theta', shape=num_radial, units='rad')

        self.add_output('normal_load_dist', shape=num_radial, units='N')
        self.add_output('circum_load_dist', shape=num_radial, units='N')

        # Create the airfoil interpolation function. Is this the best way to do
        # this? I don't know.
        jlmain.eval(f"""
            using CCBlade
            airfoil_interp = af_from_file("{af_fname}")
        """)
        self._airfoil_interp = jlmain.airfoil_interp

    def compute(self, inputs, outputs):
        # options_d = dict(self.options)
        options_d = {'B': self.options['B'], 'af': self._airfoil_interp}
        inputs_d = dict(inputs)
        outputs_d = dict(outputs)
        self._compute(options_d, inputs_d, outputs_d)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from openmdao.api import IndepVarComp, Problem

    B = 2
    radii = np.array([0.7526, 0.7928, 0.8329, 0.8731, 0.9132, 0.9586, 1.0332,
                      1.1128, 1.1925, 1.2722, 1.3519, 1.4316, 1.5114, 1.5911,
                      1.6708, 1.7505, 1.8302, 1.9099, 1.9896, 2.0693, 2.1490,
                      2.2287, 2.3084, 2.3881, 2.4678, 2.5475, 2.6273, 2.7070,
                      2.7867, 2.8661, 2.9410])
    chord = np.array([0.6270, 0.6255, 0.6231, 0.6199, 0.6165, 0.6125, 0.6054,
                      0.5973, 0.5887, 0.5794, 0.5695, 0.5590, 0.5479, 0.5362,
                      0.5240, 0.5111, 0.4977, 0.4836, 0.4689, 0.4537, 0.4379,
                      0.4214, 0.4044, 0.3867, 0.3685, 0.3497, 0.3303, 0.3103,
                      0.2897, 0.2618, 0.1920])
    theta = np.array([40.2273, 38.7657, 37.3913, 36.0981, 34.8803, 33.5899,
                      31.6400, 29.7730, 28.0952, 26.5833, 25.2155, 23.9736,
                      22.8421, 21.8075, 20.8586, 19.9855, 19.1800, 18.4347,
                      17.7434, 17.1005, 16.5013, 15.9417, 15.4179, 14.9266,
                      14.4650, 14.0306, 13.6210, 13.2343, 12.8685, 12.5233,
                      12.2138])
    num_radial = len(radii)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('rho', val=1.225, units='kg/m**3')
    comp.add_output('v', val=10., units='m/s')
    comp.add_output('omega', val=8000.*(2*np.pi/60), units='rad/s')
    comp.add_output('hub_diameter', val=1., units='inch')
    comp.add_output('prop_diameter', val=6., units='inch')
    comp.add_output('radii', val=radii, units='inch')
    comp.add_output('chord', val=chord, units='inch')
    comp.add_output('theta', val=theta, units='deg')
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = CCBladeWrapperComp(num_radial, B=B,
                              af_fname='airfoils/NACA64_A17.dat')
    prob.model.add_subsystem('ccblade_comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()

    prop_radius = 0.5*prob.get_val('prop_diameter', units='inch')
    radii = prob.get_val('radii', units='inch')/prop_radius
    normal_load_dist = prob.get_val('normal_load_dist', units='N')
    circum_load_dist = prob.get_val('circum_load_dist', units='N')

    fig, ax = plt.subplots()
    ax.plot(radii, normal_load_dist, label='normal load')
    ax.plot(radii, circum_load_dist, label='circum load')
    ax.legend()
    fig.savefig('foo.png')
