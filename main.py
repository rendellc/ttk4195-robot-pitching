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
        sim_args['speed_factor'] = 1.0
        sim_args['step_size'] = 0.01
        sim_args['print_info'] = False
        sim_args['live_plot'] = False
        sim_args['constraint_warnings'] = True
        self.sim = simulation.Simulation(**sim_args)

    def episode(self, controller_func):

        reward = -1.0
        success = False
        msg = ""
        try:
            q_pitch, t_pitch = self.sim.simulate_with_controller(
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

            ball_land, t_pitch = self.sim.simulate_with_controller(
                self.flying_ball,
                t_pitch,
                ball_throw,
                lambda *_: None,
                lambda t, x: x[1] <= 0 # ball has hit the ground
            )

            reward = ball_land[0]
            success = True
        except RuntimeError as e:
            msg = str(e)

        return reward, success, msg


def main():
    env = Environment()
    def episode_runner(controller_func):
        reward, success, info = env.episode(controller_func)
        if info:
            print(info)
        print("Got reward {0}".format(reward))

    episode_runner(lambda t,q: -15.0)
    episode_runner(lambda t,q: -25.0)
    episode_runner(lambda t,q: -35.0)


if __name__ == "__main__":
    main()
