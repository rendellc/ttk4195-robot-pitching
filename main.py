import environment
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt


def controller_template(parameters):
    p = parameters # shorter name

    controller_func = lambda t, q: p[0]*q[0] + p[1]*q[1] + p[2]*t**2 + p[3]*t + p[4]

    return controller_func

def main():
    env = environment.Environment()


    best_reward = 2.0
    best_parameters = [17,-17,18,-6,-7]
    #limits = [(-10,10), (-10,10), (-10,10), (-10,10), (-20,0)]
    limits = [(-3,17), (-22,-2), (-11,9), (-6,14), (-7,13)]

    n_episodes = 1000000
    for i in range(n_episodes):
        if i%10000 == 0:
            print(i)

        controller_func, parameters = env.action_sample(controller_template, limits)
        reward, info = env.episode(controller_func)

        p = parameters
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters

            # move limits to be centered on best_parameters
            for i in range(len(limits)):
                limits[i] = (p[i] - 10, p[i] + 10)

            print("best reward {0}, new limits {1}".format(reward, limits))

        #q_table[p[0], p[1], p[2], p[3], p[4]] = reward

    # Replay best parameter setup
    env.sim.record_path = True
    controller_func = controller_template(best_parameters)
    reward, info = env.episode(controller_func)
    pitching_path = info['pitching_path']
    flying_path = info['flying_path']

    for point in pitching_path:
        ball = env.pitch_bot.ball_states(point[1])
        (x1, y1), (x2, y2) = env.pitch_bot.joint_positions(point[1][0], point[1][1])
        plt.scatter(ball[0], ball[1], c='b')
        plt.scatter(x1, y1, c='y')
        x2c = (x2 - x1)*0.5 + x1
        y2c = (y2 - y1)*0.5 + y1
        plt.scatter(x2c,y2c, c='g')
        plt.pause(env.sim.step_size)

    for point in flying_path:
        plt.scatter(point[1][0], point[1][1], c='b')
        plt.pause(env.sim.step_size)


    plt.show()


if __name__ == "__main__":
    main()
