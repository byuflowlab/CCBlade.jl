import numpy as np
import openmdao.api as om
from omjl.julia_comps import JuliaImplicitComp
from ccblade.ccblade_py import FunctionalsComp
import julia.CCBlade as CCBlade


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

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']
        num_blades = self.options['num_blades']
        af_filename = self.options['af_filename']
        turbine = self.options['turbine']

        comp = om.ExecComp('hub_radius = 0.5*hub_diameter',
                           hub_radius={'value': 0.1, 'units': 'm'},
                           hub_diameter={'units': 'm'})
        self.add_subsystem('hub_radius_comp', comp, promotes=['*'])

        comp = om.ExecComp('prop_radius = 0.5*prop_diameter',
                           prop_radius={'value': 1.0, 'units': 'm'},
                           prop_diameter={'units': 'm'})
        self.add_subsystem('prop_radius_comp', comp, promotes=['*'])

        af = CCBlade.af_from_file(af_filename, use_interpolations_jl=True)
        julia_comp_data = CCBlade.CCBladeResidualComp(
            num_nodes=num_nodes, num_radial=num_radial, af=af, B=num_blades,
            turbine=turbine, debug_print=True)
        comp = JuliaImplicitComp(julia_comp_data=julia_comp_data)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)
        self.add_subsystem('ccblade_comp', comp,
                           promotes_inputs=[("r", "radii"), "chord", "theta",
                                            "Vx", "Vy", "rho", "mu", "asound",
                                            ("Rhub", "hub_radius"),
                                            ("Rtip", "prop_radius"),
                                            "precone"],
                           promotes_outputs=['Np', 'Tp'])

        comp = FunctionalsComp(num_nodes=num_nodes, num_radial=num_radial,
                               num_blades=num_blades)
        self.add_subsystem(
            'ccblade_torquethrust_comp', comp,
            promotes_inputs=['radii', 'dradii', 'Np', 'Tp', 'v', 'omega'],
            promotes_outputs=['thrust', 'torque', 'power', 'efficiency'])

        self.linear_solver = om.DirectSolver(assemble_jac=True)
