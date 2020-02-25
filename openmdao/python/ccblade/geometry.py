import numpy as np
from openmdao.api import ExplicitComponent, Group, SplineComp
from openmdao.utils.spline_distributions import cell_centered
from ccblade.utils import get_rows_cols


def tile_sparse_jac(data, rows, cols, nrow, ncol, num_nodes):
    nnz = len(rows)

    if np.isscalar(data):
        data = data * np.ones(nnz)

    if not np.isscalar(nrow):
        nrow = np.prod(nrow)

    if not np.isscalar(ncol):
        ncol = np.prod(ncol)

    data = np.tile(data, num_nodes)
    rows = np.tile(rows, num_nodes) + np.repeat(np.arange(num_nodes), nnz) * nrow
    cols = np.tile(cols, num_nodes) + np.repeat(np.arange(num_nodes), nnz) * ncol

    return data, rows, cols


class BEMTMeshComp(ExplicitComponent):
    """
    Component that computes the location of the radial points.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of pitch angles.')
        self.options.declare('num_radial', types=int,
                             desc='Number of points along the radius of the blade.')

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_radial = self.options['num_radial']

        t_ = np.linspace(0., 1., num_radial + 1) #** 0.5
        t = 0.5 * (t_[:-1] + t_[1:])
        dt = (t_[1:] - t_[:-1])

        self.t = np.outer(np.ones(num_nodes), t)
        self.dt = np.outer(np.ones(num_nodes), dt)

        self.add_input('hub_diameter', val=1., shape=num_nodes, units='cm')
        self.add_input('prop_diameter', val=10., shape=num_nodes, units='cm')

        self.add_output('radii_norm', shape=(num_nodes, num_radial))
        self.add_output('radii', shape=(num_nodes, num_radial), units='m')
        self.add_output('dradii', shape=(num_nodes, num_radial), units='m')

        rows_ = np.arange(num_radial)
        cols_ = np.zeros(num_radial, int)

        data = 1.
        _, rows, cols = tile_sparse_jac(1., rows_, cols_, num_radial, 1.0, num_nodes)
        self.declare_partials('radii_norm', 'hub_diameter', rows=rows, cols=cols)

        data = 1.
        _, rows, cols = tile_sparse_jac(data, rows_, cols_, num_radial, 1.0, num_nodes)
        self.declare_partials('radii_norm', 'prop_diameter', rows=rows, cols=cols)

        data = (1. - t) * 0.5e-2
        data, rows, cols = tile_sparse_jac(data, rows_, cols_, num_radial, 1.0, num_nodes)
        self.declare_partials('radii', 'hub_diameter', val=data, rows=rows, cols=cols)

        data = t * 0.5e-2
        data, rows, cols = tile_sparse_jac(data, rows_, cols_, num_radial, 1.0, num_nodes)
        self.declare_partials('radii', 'prop_diameter', val=data, rows=rows, cols=cols)

        data = -dt * 0.5e-2
        data, rows, cols = tile_sparse_jac(data, rows_, cols_, num_radial, 1.0, num_nodes)
        self.declare_partials('dradii', 'hub_diameter', val=data, rows=rows, cols=cols)

        data = dt * 0.5e-2
        data, rows, cols = tile_sparse_jac(data, rows_, cols_, num_radial, 1.0, num_nodes)
        self.declare_partials('dradii', 'prop_diameter', val=data, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        num_radial = self.options['num_radial']

        hub_radius_m = np.outer(inputs['hub_diameter'] * 0.5e-2, np.ones(num_radial))
        prop_radius_m = np.outer(inputs['prop_diameter'] * 0.5e-2, np.ones(num_radial))

        radii = hub_radius_m + self.t * (prop_radius_m - hub_radius_m)

        outputs['radii'] = radii
        outputs['radii_norm'] = radii / prop_radius_m
        outputs['dradii'] = self.dt * (prop_radius_m - hub_radius_m)

    def compute_partials(self, inputs, partials):
        num_radial = self.options['num_radial']

        hub_radius_m = np.outer(inputs['hub_diameter'], np.ones(num_radial)) * 0.5e-2
        prop_radius_m = np.outer(inputs['prop_diameter'], np.ones(num_radial)) * 0.5e-2

        partials['radii_norm', 'hub_diameter'] = (
            1. / prop_radius_m - self.t / prop_radius_m
        ).flatten() * 0.5e-2
        partials['radii_norm', 'prop_diameter'] = (
            -hub_radius_m / prop_radius_m ** 2 + self.t * hub_radius_m / prop_radius_m ** 2
        ).flatten() * 0.5e-2


class GeometryGroup(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('num_radial', types=int)
        self.options.declare('add_pitch', types=bool, default=False)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_cp = self.options['num_cp']
        num_radial = self.options['num_radial']

        comp = BEMTMeshComp(num_nodes=num_nodes, num_radial=num_radial)
        self.add_subsystem('mesh_comp', comp, promotes=['*'])

        x_cp = np.linspace(0.0, 1.0, num_cp)
        x_interp = cell_centered(num_radial)
        akima_options = {'delta_x': 0.1}
        comp = SplineComp(method='akima', interp_options=akima_options, x_cp_val=x_cp, x_interp_val=x_interp, vec_size=num_nodes)
        comp.add_spline(y_cp_name='chord_dv', y_interp_name='chord', y_units='m')
        comp.add_spline(y_cp_name='theta_dv', y_interp_name='theta', y_units='rad')
        self.add_subsystem('akima_comp', comp,
                           promotes_inputs=['chord_dv', 'theta_dv'],
                           promotes_outputs=['chord', 'theta'])
