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
        self.print_info = get_argument('print_info', False, **kwargs)
        self.live_plot = get_argument('live_plot', False, **kwargs)
        self.pause_live_plot = get_argument('pause_live_plot', False, **kwargs)
        self.constraint_warnings = get_argument('constraint_warnings', True, **kwargs)
        #self.early_stop_pitching = get_argument('early_stop_pitching', False, **kwargs)
        #self.skip_flying = get_argument('skip_flying', False, **kwargs)

    def _print(self, *args, **kwargs):
        if self.print_info:
            print(*args, **kwargs)

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
        self._print("Initial conditions:", initial_state)

        if self.live_plot:
            plt.clf()

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

            #if live_plot:
            #    plt.scatter(x_1, y_1, c='y')
            #    plt.scatter(x_2c, y_2c, c='y')
            #    plt.scatter(x_b, y_b, c='b')
            #    plt.pause(step_size/speed_factor)

            if end_func(sim_sys.t, sim_sys.y):
                break

        self._print("Simulation done at {0} seconds".format(sim_sys.t))
        self._print("Final state", sim_sys.y)

        return sim_sys.y, sim_sys.t

        #if skip_flying:
        #    return predict_length(ball_states(pitch_sys.y))

        #x_b, y_b, x_b_dot, y_b_dot = ball_states(pitch_sys.y)
        #x_0 = np.array([x_b, y_b, x_b_dot, y_b_dot])
        #t_0 = pitch_sys.t
        #flying_sys = ode(flying).set_integrator('dopri5')
        #flying_sys.set_initial_value(x_0, t_0)

        #while flying_sys.successful() and y_b >= 0:
        #    flying_sys.integrate(flying_sys.t + step_size)
        #    x_b, y_b = flying_sys.y[0], flying_sys.y[1]

        #    if live_plot:
        #        plt.scatter(x_b, y_b, c='b')
        #        plt.pause(step_size/speed_factor)

        #if pause_live_plot:
        #    plt.show()

        #return x_b

