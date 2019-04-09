import random
import numpy as np

import robot
import simulation

def predict_length(ball_state, g):
    x = ball_state[0]
    y = ball_state[1]
    xdot = ball_state[2]
    ydot = ball_state[3]

    t_impact = (ydot + np.sqrt(ydot**2 + 2*y*g))/g
    x_final = x + xdot*t_impact

    return x_final

""" Environment for running episodes with different action laws """
class Environment():
    def __init__(self):
        phys_args = dict()
        phys_args['gravity'] = 9.81
        phys_args['l1'] = 0.3
        phys_args['l1c'] = 0.2071
        phys_args['l2'] = 0.542
        phys_args['l2c'] = 0.2717
        phys_args['m1'] = 2.934
        phys_args['m2'] = 1.1022
        phys_args['mb'] = 0.064
        phys_args['I1'] = 0.2067
        phys_args['I2'] = 0.1362
        phys_args['k2'] = 14.1543
        phys_args['tau_max'] = 180.0
        phys_args['q1d_max'] = 3.787
        phys_args['power_max'] = 270.0
        phys_args['enforce_constraints'] = True
        self.pitch_bot = robot.PitchingBot(**phys_args)
        self.flying_ball = robot.FlyingBall(**phys_args)

        sim_args = {}
        #sim_args['integrator'] = 'vode'
        sim_args['speed_factor'] = 1.0
        sim_args['step_size'] = 0.01
        sim_args['print_info'] = False
        sim_args['live_plot'] = False
        sim_args['constraint_warnings'] = True
        self.sim = simulation.Simulation(**sim_args)

    def action_sample(self, controller_template, limits):
        """ Generate a random controller_func based on template and parameter limits
        :param controller_template, function giving the structure of the control law
        :param limits, list of pairs giving min/max for every parameter to controller_template
                       [(min1,max1),(min2,max2),...]
        """
        parameters = []
        for limit in limits:
            parameters.append(random.uniform(limit[0], limit[1]))
        controller_func = controller_template(parameters)
        return controller_func, parameters


    def episode(self, controller_func):
        """ Use the given controller_func and evaluate it on the system
        :param controller_func, function giving controller as function of time, states
        """

        reward = -1.0
        info = dict()
        info['success'] = False
        try:
            q_pitch, t_pitch, pitching_path = self.sim.simulate_with_controller(
                self.pitch_bot,
                0.0,
                np.array([5.0*np.pi/6.0,
                          np.pi + np.arcsin(self.pitch_bot.l1/(2*self.pitch_bot.l2)),
                          0.0,
                          0.0]),
                controller_func,
                lambda t, q: (self.pitch_bot.l1*np.cos(q[0]) + self.pitch_bot.l2*np.cos(q[1])) > 0 # ball has positive x coordinate
            )

            ball_throw = self.pitch_bot.ball_states(q_pitch)

            ball_land, t_land, flying_path = self.sim.simulate_with_controller(
                self.flying_ball,
                t_pitch,
                ball_throw,
                lambda *_: None,
                lambda t, x: x[1] <= 0 # ball has hit the ground
            )
            distance = ball_land[0]

            reward = distance
            info['success'] = True
            info['reward'] = reward
            info['pitching_path'] = pitching_path
            info['flying_path'] = flying_path
        except RuntimeError as e:
            info['error'] = str(e)

        return reward, info

