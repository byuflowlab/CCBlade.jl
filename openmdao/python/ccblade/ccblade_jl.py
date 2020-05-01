import os
import numpy as np
import openmdao.api as om
from omjl import make_component
from ccblade.ccblade_py import FunctionalsComp
from julia.CCBladeOpenMDAOExample import CCBladeResidualComp
from julia.CCBlade import af_from_files


sqa_training = np.array([0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])
zje_training = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
fb_training = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.992, 0.991, 0.988, 0.983, 0.976, 0.970, 0.963],
    [0.986, 0.982, 0.977, 0.965, 0.953, 0.940, 0.927],
    [0.979, 0.974, 0.967, 0.948, 0.929, 0.908, 0.887],
    [0.972, 0.965, 0.955, 0.932, 0.905, 0.872, 0.835],
    [0.964, 0.954, 0.943, 0.912, 0.876, 0.834, 0.786],
    [0.955, 0.943, 0.928, 0.892, 0.848, 0.801, 0.751],
    [0.948, 0.935, 0.917, 0.872, 0.820, 0.763, 0.706],
    [0.940, 0.924, 0.902, 0.848, 0.790, 0.726, 0.662]])


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


class CCBladeGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('num_blades', types=int)
        self.options.declare('af_filename', types=str)
        self.options.declare('turbine', types=bool)
        self.options.declare('installed_thrust_loss', types=bool, default=False)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        num_blades = self.options['num_blades']
        af_filename = self.options['af_filename']
        turbine = self.options['turbine']
        installed_thrust_loss = self.options['installed_thrust_loss']

        comp = om.ExecComp(
            ['hub_radius = 0.5*hub_diameter',
             'prop_radius = 0.5*prop_diameter'],
            shape=num_nodes, has_diag_partials=True,
            hub_radius={'units': 'm'},
            hub_diameter={'units': 'm'},
            prop_radius={'units': 'm'},
            prop_diameter={'units': 'm'})
        self.add_subsystem('hub_prop_radius_comp', comp, promotes=['*'])

        # Stole this from John Hwang's OpenBEMT code.
        this_dir = os.path.split(__file__)[0]
        file_path = os.path.join(this_dir, 'airfoils', af_filename)
        af = af_from_files([file_path])

        comp = make_component(CCBladeResidualComp(
            num_nodes=num_nodes, num_radial=num_radial, af=af, B=num_blades,
            turbine=turbine))
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        self.add_subsystem('ccblade_comp', comp,
                           promotes_inputs=[("r", "radii"), "chord", "theta",
                                            "Vx", "Vy", "rho", "mu", "asound",
                                            ("Rhub", "hub_radius"),
                                            ("Rtip", "prop_radius"),
                                            "precone", "pitch"],
                           promotes_outputs=['Np', 'Tp'])

        comp = FunctionalsComp(num_nodes=num_nodes, num_radial=num_radial,
                               num_blades=num_blades)
        if installed_thrust_loss:
            promotes_outputs = [('thrust', 'uninstalled_thrust'), 'torque', 'power', 'efficiency']
        else:
            promotes_outputs = ['thrust', 'torque', 'power', 'efficiency']

        self.add_subsystem(
            'ccblade_torquethrust_comp', comp,
            promotes_inputs=['hub_radius', 'prop_radius', 'radii', 'Np', 'Tp', 'v', 'omega'],
            promotes_outputs=promotes_outputs)

        if installed_thrust_loss:
            comp = BertonInstalledThrustLossInputComp(num_nodes=num_nodes)
            self.add_subsystem(
                'installed_thrust_loss_inputs_comp', comp,
                promotes_inputs=['omega', 'v', 'prop_diameter'],
                promotes_outputs=['sqa', 'zje'])

            comp = om.MetaModelStructuredComp(vec_size=num_nodes)
            comp.add_input('sqa', val=0.0, training_data=sqa_training, units=None)
            comp.add_input('zje', val=0.0, training_data=zje_training, units=None)
            comp.add_output('fb', val=1.0, training_data=fb_training, units=None)
            self.add_subsystem(
                'installed_thrust_loss_mm_comp', comp,
                promotes_inputs=['sqa', 'zje'],
                promotes_outputs=['fb'])

            comp = om.ExecComp(
                'thrust = fb*uninstalled_thrust',
                has_diag_partials=True,
                uninstalled_thrust={'units': 'N', 'shape': num_nodes},
                fb={'units': None, 'shape': num_nodes},
                thrust={'units': 'N', 'shape': num_nodes})
            self.add_subsystem('installed_thrust_loss_comp', comp,
                               promotes_inputs=['fb', 'uninstalled_thrust'],
                               promotes_outputs=['thrust'])


class BertonInstalledThrustLossInputComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('eht', desc='Engine envelope height, inches', default=15.0)
        self.options.declare('ewid', desc='Engine envelope width, inches', default=24.0)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('omega', shape=num_nodes, units='rad/s')
        self.add_input('prop_diameter', shape=num_nodes, units='ft')
        self.add_input('v', shape=num_nodes, units='ft/s')

        self.add_output('sqa', shape=num_nodes)
        self.add_output('zje', shape=num_nodes)

        rows = cols = np.arange(num_nodes)
        self.declare_partials('sqa', 'prop_diameter', rows=rows, cols=cols)
        self.declare_partials('zje', '*', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in2_ft2 = 12.0**2

        eht = self.options['eht']
        ewid = self.options['ewid']
        anac = np.pi * (8.0+eht) * (8.0+ewid) / in2_ft2 / 4.0

        omega = inputs['omega']
        v = inputs['v']
        prop_diameter = inputs['prop_diameter']

        sqa = outputs['sqa']
        zje = outputs['zje']

        prop_radius = 0.5*prop_diameter
        tip_speed = omega*prop_radius
        zji = np.pi*v/tip_speed

        sqa[:] = anac / (np.pi * prop_diameter**2 / 4.0)
        zje[:] = (1.0 - 0.254 * sqa) * zji

    def compute_partials(self, inputs, partials):
        in2_ft2 = 12.0**2

        eht = self.options['eht']
        ewid = self.options['ewid']
        anac = np.pi * (8.0+eht) * (8.0+ewid) / in2_ft2 / 4.0

        omega = inputs['omega']
        v = inputs['v']
        prop_diameter = inputs['prop_diameter']

        sqa = anac / (np.pi * prop_diameter**2 / 4.0)
        prop_radius = 0.5*prop_diameter
        tip_speed = omega*prop_radius
        zji = np.pi*v/tip_speed

        dsqa_dprop_diameter = partials['sqa', 'prop_diameter']
        dsqa_dprop_diameter[:] = -2.0 * anac / (np.pi * prop_diameter**3 / 4.0)

        dzje_dprop_diameter = partials['zje', 'prop_diameter']
        dzji_dprop_diameter = -np.pi*v/(omega*0.5*prop_diameter**2)
        dzje_dprop_diameter[:] = -0.254*dsqa_dprop_diameter*zji + (1 - 0.254*sqa)*dzji_dprop_diameter

        dzje_domega = partials['zje', 'omega']
        dzji_domega = -np.pi * v / (omega**2 * 0.5*prop_diameter)
        dzje_domega[:] = (1.0 - 0.254*sqa)*dzji_domega

        dzje_dv = partials['zje', 'v']
        dzji_dv = np.pi / (omega*0.5*prop_diameter)
        dzje_dv[:] = (1.0 - 0.254 * sqa)*dzji_dv


if __name__ == "__main__":
    nn = 5
    prob = om.Problem()

    ivc = om.IndepVarComp()
    ivc.add_output('omega', val=2700.0, shape=nn, units='rev/min')
    ivc.add_output('v', val=150.0, shape=nn, units='knot')
    ivc.add_output('prop_diameter', val=74.0, shape=nn, units='inch')
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])

    comp = BertonInstalledThrustLossInputComp(num_nodes=nn)
    prob.model.add_subsystem('thrust_loss_inputs_comp', comp, promotes=['*'])

    comp = om.MetaModelStructuredComp(vec_size=nn)
    comp.add_input('sqa', val=0.0, training_data=sqa_training, units=None)
    comp.add_input('zje', val=0.0, training_data=zje_training, units=None)
    comp.add_output('fb', val=1.0, training_data=fb_training, units=None)
    prob.model.add_subsystem(
        'installed_thrust_loss_mm_comp', comp,
        promotes_inputs=['sqa', 'zje'],
        promotes_outputs=['fb'])

    comp = om.ExecComp(
        'installed_thrust = fb*bemt_thrust',
        has_diag_partials=True,
        bemt_thrust={'units': 'N', 'shape': nn},
        fb={'units': None, 'shape': nn},
        installed_thrust={'units': 'N', 'shape': nn})
    prob.model.add_subsystem(
        'installed_thrust_loss_comp', comp,
        promotes_inputs=['fb', 'bemt_thrust'],
        promotes_outputs=['installed_thrust'])

    prob.setup()
    prob.run_model()
    print(f"sqa = {prob.get_val('sqa')}")
    print(f"zje = {prob.get_val('zje')}")
    print(f"fb = {prob.get_val('fb')}")
    prob.check_partials()
