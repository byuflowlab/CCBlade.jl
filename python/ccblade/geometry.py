import numpy as np
from openmdao.api import ExplicitComponent, AkimaSplineComp, Group


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


class BEMTThetaComp(ExplicitComponent):
    """
    Compute the pitched theta control points by subtracting pitch from the unpitched thetas.

    Unpitched thetas are of dim num_cp, and pitch angle is of dim num_nodes, so the pitched
    theta control points have dim (num_nodes, num_cp).
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of pitch angles.')
        self.options.declare('num_cp', types=int,
                             desc='Number of radial control points for chord and theta splines.')

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_cp = self.options['num_cp']

        self.add_input('theta_cp_unpitched', shape=num_cp, units='rad')
        self.add_input('pitch_cp', val=0, shape=num_nodes, units='rad')

        self.add_output('theta_cp', shape=(num_nodes, num_cp), units='rad')

        rows = np.arange(num_nodes * num_cp)
        cols = np.tile(np.arange(num_cp), num_nodes)
        self.declare_partials('theta_cp', 'theta_cp_unpitched', rows=rows, cols=cols, val=1.0)

        cols = np.repeat(np.arange(num_nodes), num_cp)
        self.declare_partials('theta_cp', 'pitch_cp', rows=rows, cols=cols, val=-1.0)

    def compute(self, inputs, outputs):
        theta_u = inputs['theta_cp_unpitched']
        pitch = inputs['pitch_cp']

        outputs['theta_cp'] = np.add.outer(-pitch, theta_u)


class GeometryGroup(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('num_radial', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        num_cp = self.options['num_cp']
        num_radial = self.options['num_radial']

        comp = BEMTMeshComp(num_nodes=num_nodes, num_radial=num_radial)
        self.add_subsystem('mesh_comp', comp, promotes=['*'])

        comp = BEMTThetaComp(num_nodes=num_nodes, num_cp=num_cp)
        self.add_subsystem('theta_cp_comp', comp,
                           promotes_inputs=[('theta_cp_unpitched', 'theta_dv'),
                                            ('pitch_cp', 'pitch')],
                           promotes_outputs=['theta_cp'])

        comp = AkimaSplineComp(vec_size=1, num_control_points=num_cp,
                               num_points=num_radial, name='chord', units='cm',
                               eval_at='cell_center')

        self.add_subsystem('chord_bspline_comp', comp,
                           promotes_inputs=[('chord:y_cp', 'chord_dv')],
                           promotes_outputs=[('chord:y', 'chord')])

        comp = AkimaSplineComp(vec_size=num_nodes, num_control_points=num_cp,
                               num_points=num_radial,
                               name='theta', units='rad',
                               eval_at='cell_center')

        self.add_subsystem('theta_bspline_comp', comp,
                           promotes_inputs=[('theta:y_cp', 'theta_cp')],
                           promotes_outputs=[('theta:y', 'theta')])
