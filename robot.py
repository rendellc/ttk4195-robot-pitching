""" Robot simulation module
Contains parameters and methods needed to simulate a ball throwing robot.
"""
import numpy as np

from util import get_argument

class RobotBase():
    def derivative(self, t, state, tau):
        """ Compute continous derivate of states """
        raise NotImplementedError("derivative function not implemented")

    #def saturate(self, state, force):
    #    """ Check for saturation and returns (saturated_state, force) """
    #    raise NotImplementedError("saturate function not implemented")

    def check_constraints(self, state, force):
        """ Check if current state and force is valid, must return true every iteration step """
        raise NotImplementedError("check_limits function not implemented")


class PitchingBot(RobotBase):
    def __init__(self, **phys_args): #g, l1, l1_c, l2, l2_c, m1, m2, mb, k2, tau_max, q1_max, power_max):
        self.g = get_argument('gravity', 9.81, **phys_args)

        # Robot specification
        self.l1 = get_argument('l1', None, **phys_args)
        self.l1c = get_argument('l1c', None, **phys_args)
        self.l2 = get_argument('l2', None, **phys_args)
        self.l2c = get_argument('l2c', None, **phys_args)
        self.m1 = get_argument('m1', None, **phys_args)
        self.m2 = get_argument('m2', None, **phys_args)
        self.mb = get_argument('mb', None, **phys_args)
        self.k2 = get_argument('k2', None, **phys_args)
        self.I1 = get_argument('I1', None, **phys_args)
        self.I2 = get_argument('I2', None, **phys_args)

        # Robot constraints
        self.tau_max = get_argument('tau_max', None, **phys_args)
        self.q1d_max = get_argument('q1d_max', None, **phys_args)
        self.power_max = get_argument('power_max', None, **phys_args)
        self.enforce_constraints = get_argument('enforce_constraints', True, **phys_args)

    def derivative(self, _, states, tau):
        """
        y' = f(t,y,tau)
        Dynamic equations for robot
        """
        result = np.zeros(4)
        result[0], result[1] = states[2], states[3]

        c_1 = np.cos(states[0])
        c_2 = np.cos(states[1])
        result[2] = (-self.m1*self.g*self.l1c*c_1 - self.m2*self.g*self.l1*c_1 -
                     self.mb*self.g*self.l1*c_1 + self.k2*(states[1] - states[0]) +
                     tau)/self.I1
        result[3] = (-self.m2*self.g*self.l2c*c_2 - self.mb*self.g*self.l2*c_2 -
                     self.k2*(states[1] - states[0]))/self.I2

        return result

    def check_constraints(self, t, q, tau):
        violated_result = False
        if not self.enforce_constraints:
            violated_result = True # Return true even if constraint was violated

        if abs(tau) > self.tau_max:
            return violated_result, "too high force"
        if abs(q[2]) > self.q1d_max:
            return violated_result, "too high q1'"
        if abs(tau*q[2]) > self.power_max:
            return violated_result, "too high power"
        if q[2] > 0:
            return violated_result, "Controller must ensure q1 is monotonically decreasing"

        return True, ""

    def joint_positions(self, q_1, q_2):
        """ Return position of joint """
        x_1 = self.l1*np.cos(q_1)
        y_1 = self.l1*np.sin(q_1)

        x_2 = x_1 + self.l2*np.cos(q_2)
        y_2 = y_1 + self.l2*np.sin(q_2)
        return (x_1, y_1), (x_2, y_2)

    def ball_states(self, q):
        """ Return the (x,y,x',y') state of the ball in pitching phase """
        c_1 = np.cos(q[0])
        s_1 = np.sin(q[0])
        c_2 = np.cos(q[1])
        s_2 = np.sin(q[1])

        x_b = self.l1*c_1 + self.l2*c_2
        y_b = self.l1*s_1 + self.l2*s_2
        x_b_dot = -self.l1*s_1*q[2] - self.l2*s_2*q[3]
        y_b_dot = self.l1*c_1*q[2] + self.l2*c_2*q[3]

        return np.array([x_b, y_b, x_b_dot, y_b_dot])


class FlyingBall(RobotBase):
    def __init__(self, **phys_args):
        self.g = get_argument('gravity', 9.81, **phys_args)

    def derivative(self, _, states, __):
        """
        states = [x,y,x',y']
        x'' = 0
        y'' = -g
        """
        return np.array([states[2], states[3], 0, -self.g])

    def check_constraints(self, *_):
        return True, ""

