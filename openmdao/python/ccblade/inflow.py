import numpy as np
import openmdao.api as om
from ccblade.utils import get_rows_cols


class SimpleInflow(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('dynamic_coloring', types=bool, default=False)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_input('v', shape=num_nodes, units='m/s')
        self.add_input('omega', shape=num_nodes, units='rad/s')
        self.add_input('radii', shape=(num_nodes, num_radial), units='m')
        self.add_input('precone', shape=num_nodes, units='rad')

        self.add_output('Vx', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('Vy', shape=(num_nodes, num_radial), units='m/s')

        if self.options['dynamic_coloring']:
            deriv_method = 'fd'
            self.declare_partials('Vx', ['v', 'precone'], method=deriv_method)
            self.declare_partials('Vy', ['omega', 'radii', 'precone'],
                                  method=deriv_method)

            self.declare_coloring(wrt='*', method=deriv_method,
                                  perturb_size=1e-5, num_full_jacs=2,
                                  tol=1e-20, orders=20, show_summary=True,
                                  show_sparsity=False)
        else:
            ss_sizes = {'i': num_nodes, 'j': num_radial}
            rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss='ij', wrt_ss='i')

            self.declare_partials('Vx', ['v', 'precone'], rows=rows, cols=cols)
            self.declare_partials('Vy', ['omega', 'precone'], rows=rows, cols=cols)

            rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss='ij', wrt_ss='ij')
            self.declare_partials('Vy', 'radii', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        v = inputs['v'][:, np.newaxis]
        omega = inputs['omega'][:, np.newaxis]
        radii = inputs['radii']
        precone = inputs['precone'][:, np.newaxis]

        Vx = outputs['Vx']
        Vy = outputs['Vy']

        Vx[:, :] = v * np.cos(precone)
        Vy[:, :] = omega * radii * np.cos(precone)

    def compute_partials(self, inputs, partials):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        v = inputs['v'][:, np.newaxis]
        omega = inputs['omega'][:, np.newaxis]
        radii = inputs['radii']
        precone = inputs['precone'][:, np.newaxis]

        dVx_dv = partials['Vx', 'v']
        dVx_dv.shape = (num_nodes, num_radial)
        dVx_dv[:, :] = np.cos(precone)
        dVx_dv.shape = (-1,)

        dVx_dprecone = partials['Vx', 'precone']
        dVx_dprecone.shape = (num_nodes, num_radial)
        dVx_dprecone[:, :] = -v * np.sin(precone)
        dVx_dprecone.shape = (-1,)

        dVy_domega = partials['Vy', 'omega']
        dVy_domega.shape = (num_nodes, num_radial)
        dVy_domega[:, :] = radii * np.cos(precone)
        dVy_domega.shape = (-1,)

        dVy_dradii = partials['Vy', 'radii']
        dVy_dradii.shape = (num_nodes, num_radial)
        dVy_dradii[:, :] = omega * np.cos(precone)
        dVy_dradii.shape = (-1,)

        dVy_dprecone = partials['Vy', 'precone']
        dVy_dprecone.shape = (num_nodes, num_radial)
        dVy_dprecone[:, :] = -omega * radii * np.sin(precone)
        dVy_dprecone.shape = (-1,)


class WindTurbineInflow(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_input('vhub', shape=num_nodes, units='m/s')
        self.add_input('omega', shape=num_nodes, units='rad/s')
        self.add_input('radii', shape=(num_nodes, num_radial), units='m')
        self.add_input('precone', shape=num_nodes, units='rad')

        # Should yaw, tilt, shear_exp have length num_nodes? Let's say yes.
        self.add_input('yaw', shape=num_nodes, units='rad')
        self.add_input('tilt', shape=num_nodes, units='rad')
        self.add_input('shear_exp', shape=num_nodes, units=None)

        # Might need an additional array dimension for the azimuthal angle some
        # day.
        self.add_input('azimuth', shape=num_nodes, units='rad')
        self.add_input('hub_height', shape=num_nodes, units='m')

        self.add_output('Vx', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('Vy', shape=(num_nodes, num_radial), units='m/s')

        self.declare_partials('Vx', ['*'])
        self.declare_partials('Vy', ['*'])

        deriv_method = 'fd'
        self.declare_coloring(wrt='*', method=deriv_method, perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs):
        vhub = inputs['vhub'][:, np.newaxis]
        omega = inputs['omega'][:, np.newaxis]
        radii = inputs['radii']
        precone = inputs['precone'][:, np.newaxis]
        yaw = inputs['yaw'][:, np.newaxis]
        tilt = inputs['tilt'][:, np.newaxis]
        azimuth = inputs['azimuth'][:, np.newaxis]
        hub_height = inputs['hub_height'][:, np.newaxis]
        shear_exp = inputs['shear_exp'][:, np.newaxis]

        Vx = outputs['Vx']
        Vy = outputs['Vy']

        sy = np.sin(yaw)
        cy = np.cos(yaw)
        st = np.sin(tilt)
        ct = np.cos(tilt)
        sa = np.sin(azimuth)
        ca = np.cos(azimuth)
        sc = np.sin(precone)
        cc = np.cos(precone)

        # coordinate in azimuthal coordinate system
        x_az = -radii*np.sin(precone)  # (num_nodes, num_radii)
        z_az = radii*np.cos(precone)  # (num_nodes, num_radii)
        y_az = 0.0  # could omit (the more general case allows for presweep so this is nonzero)

        # get section heights in wind-aligned coordinate system
        height_from_hub = (y_az*sa + z_az*ca)*ct - x_az*st  # (num_nodes, num_radii)

        # velocity with shear
        V = vhub*(1 + height_from_hub/hub_height)**shear_exp  # (num_nodes, num_radial)

        # transform wind to blade c.s.
        Vwind_x = V * ((cy*st*ca + sy*sa)*sc + cy*ct*cc)  # (num_nodes, num_radial)
        Vwind_y = V * (cy*st*sa - sy*ca)  # (num_nodes, num_radial)

        # wind from rotation to blade c.s.
        Vrot_x = -omega*y_az*sc  # (num_nodes, 1)
        Vrot_y = omega*z_az  # (num_nodes, num_radial)

        # total velocity
        Vx[:, :] = Vwind_x + Vrot_x
        Vy[:, :] = Vwind_y + Vrot_y


if __name__ == "__main__":
    num_nodes = 2
    num_blades = 3
    num_radial = 4

    v = np.random.random((num_nodes,))
    omega = np.random.random((num_nodes,))
    radii = np.random.random((num_nodes, num_radial))
    precone = np.random.random((num_nodes,))

    p = om.Problem()

    comp = om.IndepVarComp()
    comp.add_output('v', val=v, units='m/s')
    comp.add_output('omega', val=omega, units='rad/s')
    comp.add_output('radii', val=radii, units='m')
    comp.add_output('precone', val=precone, units='rad')
    p.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = SimpleInflow(
        num_nodes=num_nodes, num_radial=num_radial)
    p.model.add_subsystem('simple_inflow_comp', comp, promotes=['*'])

    p.setup()
    p.check_partials()
