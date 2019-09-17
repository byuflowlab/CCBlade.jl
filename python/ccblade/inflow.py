import numpy as np
import openmdao.api as om


class SimpleInflow(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_radial', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        self.add_input('v', shape=num_nodes, units='m/s')
        self.add_input('omega', shape=num_nodes, units='rad/s')
        self.add_input('radii', shape=(num_nodes, num_radial), units='m')
        self.add_input('precone', units='rad')

        self.add_output('Vx', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('Vy', shape=(num_nodes, num_radial), units='m/s')

        self.declare_partials('Vx', ['v', 'precone'])
        self.declare_partials('Vy', ['omega', 'radii', 'precone'])

        deriv_method = 'fd'
        self.declare_coloring(wrt='*', method=deriv_method, perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs):
        v = inputs['v']
        omega = inputs['omega']
        radii = inputs['radii']
        precone = inputs['precone']

        Vx = outputs['Vx']
        Vy = outputs['Vy']

        Vx[:, :] = v[:, np.newaxis] * np.cos(precone)
        Vy[:, :] = omega[:, np.newaxis] * radii * np.cos(precone)


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
        self.add_input('precone', units='rad')

        # Should yaw, tilt, shear_exp have length num_nodes?
        self.add_input('yaw', units='rad')
        self.add_input('tilt', units='rad')
        self.add_input('shear_exp', units=None)

        # Might need an additional array dimension for the azimuthal angle.
        self.add_input('azimuth', units='rad')
        self.add_input('hub_height', units='m')

        self.add_output('Vx', shape=(num_nodes, num_radial), units='m/s')
        self.add_output('Vy', shape=(num_nodes, num_radial), units='m/s')

        self.declare_partials('Vx', ['*'])
        self.declare_partials('Vy', ['*'])

        deriv_method = 'fd'
        self.declare_coloring(wrt='*', method=deriv_method, perturb_size=1e-5,
                              num_full_jacs=2, tol=1e-20, orders=20,
                              show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs):
        vhub = inputs['vhub']
        omega = inputs['omega']
        radii = inputs['radii']
        precone = inputs['precone']
        yaw = inputs['yaw']
        tilt = inputs['tilt']
        azimuth = inputs['azimuth']
        hub_height = inputs['hub_height']
        shear_exp = inputs['shear_exp']

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
        V = vhub[:, np.newaxis]*(1 + height_from_hub/hub_height)**shear_exp  # (num_nodes, num_radial)

        # transform wind to blade c.s.
        Vwind_x = V * ((cy*st*ca + sy*sa)*sc + cy*ct*cc)  # (num_nodes, num_radial)
        Vwind_y = V * (cy*st*sa - sy*ca)  # (num_nodes, num_radial)

        # wind from rotation to blade c.s.
        Vrot_x = -omega[:, np.newaxis]*y_az*sc  # (num_nodes, 1)
        Vrot_y = omega[:, np.newaxis]*z_az  # (num_nodes, num_radial)

        # total velocity
        Vx[:, :] = Vwind_x + Vrot_x
        Vy[:, :] = Vwind_y + Vrot_y
