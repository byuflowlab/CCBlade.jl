
import numpy as np

from openmdao.api import IndepVarComp, Problem

from openbemt.airfoils.process_airfoils import ViternaAirfoil
from ccblade.geometry import GeometryGroup
from ccblade.ccblade_py import CCBladeGroup


def make_individual_plots(prob, phi, phi_residual, fmt):
    import matplotlib.pyplot as plt

    node = 0
    num_radial = prob.model.ccblade_group.options['num_radial']
    for i in range(num_radial):
        fig, ax = plt.subplots()

        radius = prob.get_val('radii', units='m')[node, i]
        Vy = prob.get_val('ccblade_group.Vy', units='m/s')[node, i]
        Vx = prob.get_val('ccblade_group.Vx', units='m/s')[node, 0]
        hub_radius = prob.get_val('ccblade_group.hub_radius', units='m')[0]
        prop_radius = prob.get_val('ccblade_group.prop_radius', units='m')[0]
        r_nondim = (
            (radius - hub_radius)/(prop_radius - hub_radius))
        phi_guess = np.degrees(np.arctan2(Vx, Vy))

        ax.plot(np.degrees(phi[:, node, i]), phi_residual[:, node, i],
                label='R(phi)')
        ax.plot(
            [-phi_guess, -phi_guess],
            [np.min(phi_residual[:, node, i]),
             np.max(phi_residual[:, node, i])])
        # ax.set_title(
        #     f'r_nondim = {r_nondim:.3f}, phi_guess = {phi_guess:.3f} deg')
        ax.set_title(
            f'r_nondim = {r_nondim:3f}, phi_guess = {phi_guess:3f} deg')
        ax.grid()
        ax.legend()
        fig.savefig(fmt.format(i))
        plt.close(fig)


def main():
    interp = ViternaAirfoil().create_akima(
        'mh117', Re_scaling=False, extend_alpha=True)

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

    num_nodes = 1
    num_blades = 3
    num_radial = 15
    num_cp = 6
    chord = 10.
    theta = np.linspace(65., 25., num_cp)*np.pi/180.
    pitch = 0.

    hub_diameter = 30.  # cm
    prop_diameter = 150.  # cm
    c0 = np.sqrt(1.4*287.058*300.)  # meters/second
    rho0 = 1.4*98600./(c0*c0)  # kg/m^3
    omega = 236.

    prob = Problem()

    comp = IndepVarComp()
    comp.add_discrete_input('B', val=num_blades)
    comp.add_output('rho', val=rho0, shape=num_nodes, units='kg/m**3')
    comp.add_output('mu', val=1., shape=num_nodes, units='N/m**2*s')
    comp.add_output('asound', val=c0, shape=num_nodes, units='m/s')
    comp.add_output('v', val=77.2, shape=num_nodes, units='m/s')
    comp.add_output('alpha', val=0., shape=num_nodes, units='rad')
    comp.add_output('incidence', val=0., shape=num_nodes, units='rad')
    comp.add_output('precone', val=0., units='deg')
    comp.add_output('omega', val=omega, shape=num_nodes, units='rad/s')
    comp.add_output('hub_diameter', val=hub_diameter, shape=num_nodes, units='cm')
    comp.add_output('prop_diameter', val=prop_diameter, shape=num_nodes, units='cm')
    comp.add_output('pitch', val=pitch, shape=num_nodes, units='rad')
    comp.add_output('chord_dv', val=chord, shape=num_cp, units='cm')
    comp.add_output('theta_dv', val=theta, shape=num_cp, units='rad')
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = GeometryGroup(num_nodes=num_nodes, num_cp=num_cp,
                         num_radial=num_radial)
    prob.model.add_subsystem(
        'geometry_group', comp,
        promotes_inputs=['hub_diameter', 'prop_diameter', 'chord_dv',
                         'theta_dv', 'pitch'],
        promotes_outputs=['radii', 'dradii', 'chord', 'theta'])

    comp = CCBladeGroup(num_nodes=num_nodes, num_radial=num_radial,
                        airfoil_interp=ccblade_interp,
                        turbine=False, phi_residual_solve_nonlinear=False)
    prob.model.add_subsystem(
        'ccblade_group', comp,
        promotes_inputs=['B', 'radii', 'dradii', 'chord', 'theta', 'rho', 'mu',
                         'asound', 'v', 'precone', 'omega', 'hub_diameter',
                         'prop_diameter'],
        promotes_outputs=[('Np', 'ccblade_normal_load'),
                          ('Tp', 'ccblade_circum_load')])

    prob.setup()
    prob.final_setup()

    eps = 1e-2
    num_phi = 45
    phi = np.linspace(-0.5*np.pi+eps, 0.0-eps, num_phi)
    phi = np.tile(phi[:, np.newaxis, np.newaxis], (1, num_nodes, num_radial))
    phi_residual = np.zeros_like(phi)

    for i in range(num_phi):
        p = phi[i, :, :]

        prob.set_val('ccblade_group.ccblade_comp.phi', p, units='rad')
        prob.run_model()
        prob.model.run_apply_nonlinear()
        inputs, outputs, residuals = prob.model.get_nonlinear_vectors()
        phi_residual[i, :, :] = residuals['ccblade_group.ccblade_comp.phi']

    make_individual_plots(
        prob, phi, phi_residual, 'phi_residual-r{:02d}.png')


if __name__ == "__main__":
    main()
