""" Code for simulating pitching robot """
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

from util import get_argument

class Simulation():
    def __init__(self, **kwargs):
        self.integrator = get_argument('integrator', 'dopri5', **kwargs)
        self.speed_factor = get_argument('speed_factor', 1.0, **kwargs)
        self.step_size = get_argument('step_size', 0.1, **kwargs)
        self.constraint_warnings = get_argument('constraint_warnings', True, **kwargs)
        self.record_path = get_argument('record_path', False, **kwargs)

    def simulate_with_controller(self, robot, initial_time, initial_state,
                                 controller_func, end_func):
        """ Generator for simulating the ball throwing
        :param robot, robot to simulate, should be derived from robot.RobotBase
        :param initial_time, initial time to start simulation at
        :param initial_state, initial state to start simulation at
        :param controller_func, function giving controller as function of time, states
        :param valid_check, check if numerical solution is valid, must return true on every iteration
        :param end_func, function giving true/false as function of time, states, simulation stops when end_func(states) returns true
        """
        sim_sys = ode(robot.derivative).set_integrator(self.integrator)
        sim_sys.set_initial_value(initial_state, initial_time)

        path = [[sim_sys.t, sim_sys.y]]

        while sim_sys.successful():
            # Compute force and integrate
            force = controller_func(sim_sys.t, sim_sys.y)

            # Check that force doesn't violate robot constraints
            valid, msg = robot.check_constraints(sim_sys.t, sim_sys.y, force)
            if not valid:
                raise RuntimeError("robot contraints violated: {0}".format(msg))
            if len(msg) > 0 and self.constraint_warnings:
                print("robot contraints violated: {0}".format(msg))

            # Apply step
            sim_sys.set_f_params(force)
            sim_sys.integrate(sim_sys.t + self.step_size)

            # Record path
            if self.record_path:
                path.append([sim_sys.t, sim_sys.y])

            if end_func(sim_sys.t, sim_sys.y):
                break

        return sim_sys.y, sim_sys.t, path

