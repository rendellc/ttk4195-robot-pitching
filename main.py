import environment
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import json


def controller_template(parameters):
    p = parameters # shorter name

    # desired q1 path
    q1 = lambda t: p[3]*t**3 + p[2]*t**2 + p[1]*t + p[0]
    q1dot = lambda t: 3*p[3]*t**2 + 2*p[2]*t + p[1]

    # Use the two last parameters as controller gains
    controller_func = lambda t, q: p[~1]*(q1(t) - q[0]) + p[~0]*(q1dot(t) - q[2])

    return controller_func

def main():
    env = environment.Environment()

    best_reward = 0.0
    best_parameters = [0,0,0,0,0]
    limits = [(-10,10), (-10,10), (-10,10), (0,10), (0,10)]
    with open('results.json', 'r') as infile:
        data = json.load(infile)
        best_reward = data.get('reward', best_reward)
        best_parameters = data.get('parameters', best_parameters)
        p = best_parameters
        for i in range(len(limits)):
            limits[i] = (p[i] - 10, p[i] + 10)

    n_episodes = 100000
    n_failed = 0
    for i in range(n_episodes):
        if i%1000 == 0:
            print(i, n_failed)

        controller_func, parameters = env.action_sample(controller_template, limits)
        reward, info = env.episode(controller_func)
        if reward < 0:
            n_failed += 1

        p = parameters
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters

            # move limits to be centered on best_parameters
            for i in range(len(limits)):
                limits[i] = (p[i] - 10, p[i] + 10)

            print("best reward {0}, new limits {1}".format(reward, limits))
            best = dict()
            best['reward'] = best_reward
            best['parameters'] = best_parameters
            with open('results.json','w') as outfile:
                json.dump(best, outfile)

    # Replay best parameter setup with plotting
    env.sim.record_path = True
    controller_func = controller_template(best_parameters)
    reward, info = env.episode(controller_func)
    pitching_path = info['pitching_path']
    flying_path = info['flying_path']

    #plt.axis([-1,1,0,2])
    for point in pitching_path:
        ball = env.pitch_bot.ball_states(point[1])
        (x1, y1), (x2, y2) = env.pitch_bot.joint_positions(point[1][0], point[1][1])
        plt.scatter(ball[0], ball[1], c='b')
        plt.scatter(x1, y1, c='y')
        plt.pause(env.sim.step_size)

    for point in flying_path:
        plt.scatter(point[1][0], point[1][1], c='g')
        plt.pause(env.sim.step_size)


    plt.show()


if __name__ == "__main__":
    main()
