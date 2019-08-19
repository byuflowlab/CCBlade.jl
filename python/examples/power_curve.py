import numpy as np

import openmdao.api as om

from ccblade import ccblade_py as ccb
from ccblade.geometry import GeometryGroup

from openbemt.airfoils.process_airfoils import ViternaAirfoil

interp = ViternaAirfoil().create_akima(
        'mh117', Re_scaling=False, extend_alpha=True)

# dity hack to make the OpenBEMT interpolant look like CCblade wants it
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


class Omega(om.ExplicitComponent): 

    def initialize(self): 
        self.options.declare('num_nodes', types=int)

    def setup(self): 
        num_nodes = self.options['num_nodes']

        self.add_input('v', shape=num_nodes, units='m/s')
        self.add_input('tsr2')
        self.add_input('prop_diameter')
        self.add_input('omega_max', val=500, units='rad/s')
        self.add_input('omega_min', val=200, units='rad/s')

        self.add_output('omega', shape=num_nodes, units='rad/s')


        self.declare_partials('*', '*')

        # turn on dynamic partial coloring
        self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs): 

        radius = inputs['prop_diameter']/2.

        omega = inputs['v']*inputs['tsr2']/radius

        o_max = inputs['omega_max']
        o_min = inputs['omega_min']
        omega[omega>o_max] = o_max
        omega[omega<o_min] = o_min

        outputs['omega'] = omega


class PitchBalance(om.ImplicitComponent): 

    def initialize(self): 
        self.options.declare('num_nodes', types=int)

    def setup(self): 
        self.add_input('P_rated', units='kW')

################################################
# Set up the OpenMDAO Problem
################################################


NUM_NODES = 5
NUM_RADIAL = 15
NUM_CP=6

p = om.Problem()

comp = om.IndepVarComp()
# comp.add_discrete_output('B', val=5)
comp.add_output('rho', val=1.225, shape=NUM_NODES, units='kg/m**3')
comp.add_output('mu', val=1., shape=NUM_NODES, units='N/m**2*s')
comp.add_output('asound', val=220, shape=NUM_NODES, units='m/s')
comp.add_output('v', val=np.linspace(50,200,NUM_NODES), shape=NUM_NODES, units='m/s')
comp.add_output('alpha', val=0., shape=NUM_NODES, units='rad')
comp.add_output('incidence', val=0., shape=NUM_NODES, units='rad')
comp.add_output('precone', val=0., units='deg')
# comp.add_output('omega', val=236, shape=NUM_NODES, units='rad/s')
comp.add_output('hub_diameter', val=30, units='cm')
comp.add_output('prop_diameter', val=150, units='cm')
comp.add_output('pitch', val=0, shape=NUM_NODES, units='rad')
comp.add_output('chord_dv', val=10, shape=NUM_CP, units='cm')
comp.add_output('theta_dv', val=np.linspace(65., 25., NUM_CP)*np.pi/180., 
                shape=NUM_CP, units='rad')

comp.add_output('omega_max', val=500, units='rad/s')
comp.add_output('omega_min', val=200, units='rad/s')
comp.add_output('tsr2', val=200)

p.model.add_subsystem('ivc', comp, promotes=['*'])

comp = GeometryGroup(num_nodes=NUM_NODES, num_cp=NUM_CP,
                     num_radial=NUM_RADIAL)
p.model.add_subsystem('geom', comp,
                      promotes_inputs=['chord_dv', 'theta_dv', 'pitch'],
                      promotes_outputs=['radii', 'dradii', 'chord', 'theta']
                      )

p.model.connect('hub_diameter', 'geom.hub_diameter', src_indices=[0]*NUM_NODES)
p.model.connect('prop_diameter', 'geom.prop_diameter', src_indices=[0]*NUM_NODES)


comp = ccb.CCBladeGroup(num_nodes=NUM_NODES, num_radial=NUM_RADIAL,
                        airfoil_interp=ccblade_interp,
                        turbine=False)

p.model.add_subsystem('omega_calc', Omega(num_nodes=NUM_NODES), 
                      promotes_inputs=['prop_diameter', 'omega_min', 'omega_max', 'tsr2', 'v'],
                      promotes_outputs=['omega'])

p.model.add_subsystem('ccblade', comp,
                         promotes_inputs=['B', 'radii', 'dradii', 'chord', 'theta', 'rho', 'mu',
                                          'asound', 'v', 'precone', 'omega', 'hub_diameter',
                                          'prop_diameter'],
                         promotes_outputs=[('Np', 'ccblade_normal_load'),
                                           ('Tp', 'ccblade_circum_load')])

p.setup()
p.run_model()

p.model.list_outputs(prom_name=True, print_arrays=True)

