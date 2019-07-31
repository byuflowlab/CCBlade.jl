import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import (IndepVarComp, Problem, Group, BalanceComp,
                          DirectSolver, NewtonSolver)
from ccblade_comp import CCBladeGroup


def make_plots(prob):
    import matplotlib.pyplot as plt

    node = 0
    num_blades = prob.model.ccblade_group.ccblade_comp.options['B']
    radii = prob.get_val(
        'ccblade_group.radii', units='m')[node, :]
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
    num_blades = 10
    num_radial = 15
    num_cp = 6

    af_filename = 'airfoils/mh117.dat'
    chord = 20.
    theta = 5.0*np.pi/180.0
    pitch = 0.

    # Numbers taken from the Aviary group's study of the RVLT tiltwing
    # turboelectric concept vehicle.
    n_props = 4
    hub_diameter = 30.  # cm
    prop_diameter = 15*30.48  # 15 ft in cm
    c0 = np.sqrt(1.4*287.058*300.)  # meters/second
    rho0 = 1.4*98600./(c0*c0)  # kg/m^3
    omega = 236.  # rad/s

    # Find the thrust per rotor from the vehicle's mass.
    m_full = 6367  # kg
    g = 9.81  # m/s**2
    thrust_vtol = m_full*g/n_props

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('rho', val=rho0, shape=num_nodes, units='kg/m**3')
    comp.add_output('mu', val=1., shape=num_nodes, units='N/m**2*s')
    comp.add_output('asound', val=c0, shape=num_nodes, units='m/s')
    comp.add_output('v', val=2., shape=num_nodes, units='m/s')
    comp.add_output('alpha', val=0., shape=num_nodes, units='rad')
    comp.add_output('incidence', val=0., shape=num_nodes, units='rad')
    comp.add_output('precone', val=0., units='deg')
    comp.add_output('hub_diameter', val=hub_diameter, shape=num_nodes, units='cm')
    comp.add_output('prop_diameter', val=prop_diameter, shape=num_nodes, units='cm')
    comp.add_output('pitch', val=pitch, shape=num_nodes, units='rad')
    comp.add_output('chord_dv', val=chord, shape=num_cp, units='cm')
    comp.add_output('theta_dv', val=theta, shape=num_cp, units='rad')
    comp.add_output('thrust_vtol', val=thrust_vtol, shape=num_nodes, units='N')
    prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

    balance_group = Group()

    comp = CCBladeGroup(num_nodes=num_nodes, num_radial=num_radial,
                        num_cp=num_cp, num_blades=num_blades,
                        af_filename=af_filename)
    balance_group.add_subsystem(
        'ccblade_group', comp,
        promotes_inputs=['chord_dv', 'theta_dv', 'rho', 'mu', 'asound', 'v',
                         'precone', 'omega', 'hub_diameter', 'prop_diameter',
                         'pitch'],
        promotes_outputs=['thrust', 'torque', 'efficiency'])

    comp = BalanceComp()
    comp.add_balance(
        name='omega',
        eq_units='N', lhs_name='thrust', rhs_name='thrust_vtol',
        val=omega, units='rad/s')
    balance_group.add_subsystem('thrust_balance_comp', comp, promotes=['*'])

    balance_group.linear_solver = DirectSolver(assemble_jac=True)
    balance_group.nonlinear_solver = NewtonSolver(maxiter=100, iprint=2)
    balance_group.nonlinear_solver.options['solve_subsystems'] = True
    balance_group.nonlinear_solver.options['atol'] = 1e-9

    prob.model.add_subsystem('thrust_balance_group', balance_group,
                             promotes=['*'])

    prob.setup()
    prob.final_setup()

    # Calculate the induced axial velocity at the rotor for hover, used for
    # non-diminsionalation.
    rho = prob.get_val('rho', units='kg/m**3')[0]
    hub_diameter = prob.get_val('hub_diameter', units='m')[0]
    prop_diameter = prob.get_val('prop_diameter', units='m')[0]
    thrust_vtol = prob.get_val('thrust_vtol', units='N')[0]
    A_rotor = 0.25*np.pi*(prop_diameter**2 - hub_diameter**2)
    v_h = np.sqrt(thrust_vtol/(2*rho*A_rotor))

    # Climb:
    climb_velocity_nondim = np.linspace(0.1, 2., 10)
    induced_velocity_nondim = np.zeros_like(climb_velocity_nondim)
    for vc, vi in np.nditer(
            [climb_velocity_nondim, induced_velocity_nondim],
            op_flags=[['readonly'], ['writeonly']]):

        # Run the model with the requested climb velocity.
        prob.set_val('v', vc*v_h, units='m/s')
        print(f"v = {prob.get_val('v', units='m/s')}")
        prob.run_model()

        # Calculate the area-weighted average induced velocity at the rotor.
        # Need the area of each blade section.
        radii = prob.get_val('ccblade_group.radii',
                             units='m')
        dradii = prob.get_val('ccblade_group.dradii',
                              units='m')
        dArea = 2*np.pi*radii*dradii

        # Get the induced velocity at the rotor plane for each blade section.
        Vx = prob.get_val('ccblade_group.Vx', units='m/s')
        a = prob.get_val('ccblade_group.ccblade_comp.a')

        # Get the area-weighted average of the induced velocity.
        vi[...] = np.sum(a*Vx*dArea/A_rotor)/v_h

    # Induced velocity from plain old momentum theory (for climb).
    induced_velocity_mt = (
        -0.5*climb_velocity_nondim + np.sqrt((0.5*climb_velocity_nondim)**2 + 1.))

    fig, ax = plt.subplots()
    ax.plot(climb_velocity_nondim, -induced_velocity_nondim, label='CCBlade.jl')
    ax.plot(climb_velocity_nondim, induced_velocity_mt, label='Momentum Theory')

    # Descent:
    climb_velocity_nondim = np.linspace(-4., -2., 10)
    induced_velocity_nondim = np.zeros_like(climb_velocity_nondim)
    for vc, vi in np.nditer(
            [climb_velocity_nondim, induced_velocity_nondim],
            op_flags=[['readonly'], ['writeonly']]):

        # Run the model with the requested climb velocity.
        prob.set_val('v', vc*v_h, units='m/s')
        print(f"vc = {vc}, v = {prob.get_val('v', units='m/s')}")
        prob.run_model()

        # Calculate the area-weighted average induced velocity at the rotor.
        # Need the area of each blade section.
        radii = prob.get_val('ccblade_group.radii',
                             units='m')
        dradii = prob.get_val('ccblade_group.dradii',
                              units='m')
        dArea = 2*np.pi*radii*dradii

        # Get the induced velocity at the rotor plane for each blade section.
        Vx = prob.get_val('ccblade_group.Vx', units='m/s')
        a = prob.get_val('ccblade_group.ccblade_comp.a')

        # Get the area-weighted average of the induced velocity.
        vi[...] = np.sum(a*Vx*dArea/A_rotor)/v_h

    # Induced velocity from plain old momentum theory (for descent).
    induced_velocity_mt = (
        -0.5*climb_velocity_nondim - np.sqrt((0.5*climb_velocity_nondim)**2 - 1.))

    ax.plot(climb_velocity_nondim, -induced_velocity_nondim, label='CCBlade.jl')
    ax.plot(climb_velocity_nondim, induced_velocity_mt, label='Momentum Theory')
    fig.savefig('induced_velocity.png')


if __name__ == "__main__":
    main()
