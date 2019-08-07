import time
import numpy as np

from openmdao.api import IndepVarComp, Problem, pyOptSparseDriver
from ccblade.geometry import GeometryGroup
from ccblade.ccblade import CCBladeGroup


def make_plots(prob):
    import matplotlib.pyplot as plt

    node = 0
    num_blades = prob.model.ccblade_group.ccblade_comp.options['B']
    radii = prob.get_val('radii', units='m')[node, :]
    ccblade_normal_load = prob.get_val(
        'ccblade_group.Np', units='N/m')[node, :]*num_blades
    ccblade_circum_load = prob.get_val(
        'ccblade_group.Tp', units='N/m')[node, :]*num_blades

    fig, ax = plt.subplots()
    ax.plot(radii, ccblade_normal_load, label='CCBlade.jl')
    ax.set_xlabel('blade element radius, m')
    ax.set_ylabel('normal load, N/m')
    ax.legend()
    fig.savefig('ccblade_normal_load.png')

    fig, ax = plt.subplots()
    ax.plot(radii, ccblade_circum_load, label='CCBlade.jl')
    ax.set_xlabel('blade element radius, m')
    ax.set_ylabel('circumferential load, N/m')
    ax.legend()
    fig.savefig('ccblade_circum_load.png')


def main():

    num_nodes = 1
    num_blades = 3
    num_radial = 15
    num_cp = 6
    af_filename = 'airfoils/mh117.dat'
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
    prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

    comp = GeometryGroup(num_nodes=num_nodes, num_cp=num_cp,
                         num_radial=num_radial)
    prob.model.add_subsystem(
        'geometry_group', comp,
        promotes_inputs=['hub_diameter', 'prop_diameter', 'chord_dv',
                         'theta_dv', 'pitch'],
        promotes_outputs=['radii', 'dradii', 'chord', 'theta'])

    comp = CCBladeGroup(num_nodes=num_nodes, num_radial=num_radial,
                        num_blades=num_blades, af_filename=af_filename)
    prob.model.add_subsystem(
        'ccblade_group', comp,
        promotes_inputs=['radii', 'dradii', 'chord', 'theta', 'rho', 'mu',
                         'asound', 'v', 'precone', 'omega', 'hub_diameter',
                         'prop_diameter'],
        promotes_outputs=['thrust', 'torque', 'efficiency'])

    prob.model.add_design_var('chord_dv', lower=1., upper=20.,
                              scaler=5e-2)
    prob.model.add_design_var('theta_dv',
                              lower=20.*np.pi/180., upper=90*np.pi/180.)

    prob.model.add_objective('efficiency', scaler=-1.,)
    prob.model.add_constraint('thrust', equals=700., scaler=1e-3,
                              indices=np.arange(num_nodes))
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'

    prob.setup()
    prob.final_setup()
    st = time.time()
    prob.run_driver()
    print(f"time = {time.time() - st}")
    thrust = prob.get_val('thrust', units='N')
    torque = prob.get_val('torque', units='N*m')
    efficiency = prob.get_val('efficiency')
    print(f"thrust = {thrust}")
    print(f"torque = {torque}")
    print(f"efficiency = {efficiency}")

    make_plots(prob)


if __name__ == "__main__":
    main()
