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
    #controller_func = lambda t, q: p[~1]*(q1(t) - q[0]) + p[~0]*(q1dot(t) - q[2])
    controller_func = lambda t, q: 8*(q1(t) - q[0]) + 24*(q1dot(t) - q[2])

    return controller_func

def load_results(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)
        best_reward = data.get('reward', 0.0)
        best_parameters = data.get('parameters', [0,0,0,0,0,0])
        
        return best_reward, best_parameters
    
def save_results(filename, reward, parameters):
    data = dict()
    data['reward'] = reward
    data['parameters'] = parameters
    with open(filename,'w') as outfile:
        json.dump(data, outfile)

def create_limits(parameters, radius):
    limits = []
    for p in parameters:
        limits.append((p - radius, p + radius))
    
    return limits

def main():
    env = environment.Environment()
    
    best_reward, best_parameters = load_results('results.json')
    limits = create_limits(best_parameters, 20)

    n_episodes = 100000
    n_failed = 0
    i = 0
    while True:
        i += 1
        if i%10000 == 0:
            print(i, n_failed/10000)
            n_failed = 0

        controller_func, parameters = env.action_sample(controller_template, limits)
        reward, info = env.episode(controller_func)
        if reward < 0:
            n_failed += 1

        p = parameters
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters
            
            file_best, file_parameters = load_results('results.json')
            if file_best > best_reward:
                # Use file parameters instead and keep searching
                best_reward = file_best
                best_parameters = file_parameters
            else:
                # Update file since we have found better parameters
                save_results('results.json', best_reward, best_parameters)

            # move limits to be centered on best_parameters
            limits = create_limits(best_parameters, 20)

            print("best reward {0}, new limits {1}".format(best_reward, limits))

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
