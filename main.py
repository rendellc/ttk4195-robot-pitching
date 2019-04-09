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

    Kp, Kd = 50, 15
    def controller_func(t,q):
        e = q1(t) - q[0]
        edot = q1dot(t) - q[2]
        tau = Kp*e + Kd*edot

        # limit force
        tau = min(max(tau, -180), 180)

        return tau

    return controller_func

def main():
    env = environment.Environment()

    best_reward = 0.0
    best_parameters = [0, 0, 0, 0]
    limits = [0, 0, 0, 0]
    with open('results.json', 'r') as infile:
        data = json.load(infile)
        best_reward = data.get('reward', best_reward)
        best_parameters = data.get('parameters', best_parameters)
        p = best_parameters
        limits = []
        for i in range(len(p)):
            limits.append((p[i] - 0.5, p[i] + 0.5))

    n_episodes = 100
    n_failed = 0
    for i in range(n_episodes):
        if i%10000 == 0 and i != 0:
            print(i, n_failed/i)

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
                limits[i] = (p[i] - 1, p[i] + 1)

            print("best reward {0}, new limits {1}".format(reward, limits))
            best = {
                'reward': best_reward,
                'parameters': best_parameters
            }
            with open('results.json','w') as outfile:
                json.dump(best, outfile)

    # Replay best parameter setup with plotting
    env.sim.record_path = True
    env.pitch_bot.enforce_constraints = False
    controller_func = controller_template(best_parameters)
    reward, info = env.episode(controller_func)
    pitching_path = info['pitching_path']
    flying_path = info['flying_path']

    t = []
    q1 = []
    q2 = []
    xb = []
    yb = []

    #plt.axis([-1,1,0,2])
    for point in pitching_path:
        ball = env.pitch_bot.ball_states(point[1])
        (x1, y1), (x2, y2) = env.pitch_bot.joint_positions(point[1][0], point[1][1])

        # save data for animation
        t.append(point[0])
        q1.append(point[1][0])
        q2.append(point[1][1])
        xb.append(ball[0])
        yb.append(ball[1])

        # live plot
        plt.scatter(ball[0], ball[1], c='b')
        plt.scatter(x1, y1, c='y')
        plt.pause(env.sim.step_size)

    for point in flying_path:
        # save data for animation
        t.append(point[0])
        xb.append(point[1][0])
        yb.append(point[1][1])

        # live plot
        plt.scatter(point[1][0], point[1][1], c='g')
        plt.pause(env.sim.step_size)

    paths = {}
    paths['time'] = t
    paths['q1'] = q1
    paths['q2'] = q2
    paths['xb'] = xb
    paths['yb'] = yb
    with open('paths.json', 'w') as file:
        json.dump(paths, file)

    plt.show()


if __name__ == "__main__":
    main()
