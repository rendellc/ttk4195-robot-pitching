""" Code for simulating pitching robot """
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt


class PitchingSim():
    def __init__(self, g, l1, l1_c, l2, l2_c, m1, m2, mb, k2, tau_max, q1_max, power_max):
        G = 9.81
        L1, L_1C = 0.3, 0.2071
        L2, L_2C = 0.542, 0.2717
        M1, M2, MB = 2.934, 1.1022, 0.064
        I1, I2 = 0.2067, 0.1362
        K2 = 14.1543

        TAU_MAX = 180.0
        Q1_MAX = 3.787
        POWER_MAX = 270.0



def pitching(_, states, tau):
    """
    y' = f(t,y,tau)
    Dynamic equations for robot
    """
    result = np.zeros(4)
    result[0], result[1] = states[2], states[3]

    c_1 = np.cos(states[0])
    c_2 = np.cos(states[1])
    tau_sat = max(min(tau, TAU_MAX), -TAU_MAX)
    result[2] = (-M1*G*L_1C*c_1 - M2*G*L1*c_1 -
                 MB*G*L1*c_1 + K2*(states[1] - states[0]) +
                 tau_sat)/I1
    result[3] = (-M2*G*L_2C*c_2 - MB*G*L2*c_2 -
                 K2*(states[1] - states[0]))/I2

    return result

def flying(_, states):
    """
    states = [x,y,x',y']
    x'' = 0
    y'' = -G
    """
    return np.array([states[2], states[3], 0, -G])


def joint_position(q_1):
    """ Return position of joint """
    x_1 = L1*np.cos(q_1)
    y_1 = L1*np.sin(q_1)
    return (x_1, y_1)

def joint_2com_position(q_1, q_2):
    x_1, y_1 = joint_position(q_1)
    x_2c = x_1 + L_2C*np.cos(q_2)
    y_2c = y_1 + L_2C*np.sin(q_2)

    return (x_2c, y_2c)

def ball_states(q):
    """ Return the (x,y,x',y') state of the ball in pitching phase """
    c_1 = np.cos(q[0])
    s_1 = np.sin(q[0])
    c_2 = np.cos(q[1])
    s_2 = np.sin(q[1])

    x_b = L1*c_1 + L2*c_2
    y_b = L1*s_1 + L2*s_2
    x_b_dot = -L1*s_1*q[2] - L2*s_2*q[3]
    y_b_dot = L1*c_1*q[2] + L2*c_2*q[3]

    return np.array([x_b, y_b, x_b_dot, y_b_dot])

def predict_length(ball_state):
    x = ball_state[0]
    y = ball_state[1]
    xdot = ball_state[2]
    ydot = ball_state[3]

    t_impact = (ydot + np.sqrt(ydot**2 + 2*y*G))/G
    x_final = x + xdot*t_impact

    return x_final


def get_argument(name, default, **kwargs):
    value = default
    if name in kwargs:
        value = kwargs[name]
    return value

def simulate_with_controller(tau_func, **kwargs):
    """ Generator for simulating the ball throwing
    :param tau_func, function giving tau as function of time, (q1,q2), (qd1,qd2)
    """
    def _print(*args, **kwargs):
        if print_info:
            print(*args, **kwargs)

    speed_factor = get_argument('speed_factor', 1.0, **kwargs)
    step_size = get_argument('step_size', 0.1, **kwargs)
    print_info = get_argument('print_info', False, **kwargs)
    live_plot = get_argument('live_plot', False, **kwargs)
    pause_live_plot = get_argument('pause_live_plot', False, **kwargs)
    early_stop_pitching = get_argument('early_stop_pitching', False, **kwargs)
    skip_flying = get_argument('skip_flying', False, **kwargs)

    q_0 = np.array([5.0*np.pi/6.0, np.pi + np.arcsin(L1/(2*L2)), 0.0, 0.0])
    t_0 = 0.0
    pitch_sys = ode(pitching).set_integrator('dopri5')
    pitch_sys.set_initial_value(q_0, t_0)
    _print("Initial conditions:", q_0)

    plt.clf()
    x_b, y_b, _, _ = ball_states(pitch_sys.y)
    while pitch_sys.successful() and x_b < 0.0 and pitch_sys.t < 10:
        tau = tau_func(pitch_sys.t, pitch_sys.y[:2], pitch_sys.y[2:])

        # Limit power
        if abs(pitch_sys.y[2]) > 0.1:
            tau_max = abs(POWER_MAX/pitch_sys.y[2])
            tau = max(min(tau, tau_max), -tau_max)

        pitch_sys.set_f_params(tau)
        pitch_sys.integrate(pitch_sys.t + step_size)
        # Limit q1'
        pitch_sys.y[2] = max(min(pitch_sys.y[2], Q1_MAX), -Q1_MAX)

        # Controller must ensure q1 is monotonically decreasing
        if pitch_sys.y[2] > 0:
            raise RuntimeError("Controller must ensure monotonic motion")

        x_1, y_1 = joint_position(pitch_sys.y[0])
        x_2c, y_2c = joint_2com_position(pitch_sys.y[0], pitch_sys.y[1])
        x_b, y_b, _, _ = ball_states(pitch_sys.y)

        if live_plot:
            plt.scatter(x_1, y_1, c='y')
            plt.scatter(x_2c, y_2c, c='y')
            plt.scatter(x_b, y_b, c='b')
            plt.pause(step_size/speed_factor)

    _print("Pitching done at {0} seconds".format(pitch_sys.t))
    _print("Pitch final state", pitch_sys.y)
    _print("Ball flying initial state", ball_states(pitch_sys.y))

    if skip_flying:
        return predict_length(ball_states(pitch_sys.y))

    x_b, y_b, x_b_dot, y_b_dot = ball_states(pitch_sys.y)
    x_0 = np.array([x_b, y_b, x_b_dot, y_b_dot])
    t_0 = pitch_sys.t
    flying_sys = ode(flying).set_integrator('dopri5')
    flying_sys.set_initial_value(x_0, t_0)

    while flying_sys.successful() and y_b >= 0:
        flying_sys.integrate(flying_sys.t + step_size)
        x_b, y_b = flying_sys.y[0], flying_sys.y[1]

        if live_plot:
            plt.scatter(x_b, y_b, c='b')
            plt.pause(step_size/speed_factor)

    if pause_live_plot:
        plt.show()

    return x_b

