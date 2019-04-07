import simulation as sim
import numpy as np


def main():
    """ Main function for module """
    sim_args = {}
    sim_args['speed_factor'] = 1.0
    sim_args['step_size'] = 0.01
    #sim_args['print_info'] = False
    sim_args['live_plot'] = False
    sim_args['early_stop_pitching'] = True
    sim_args['skip_flying'] = True

    best_distance = -float('inf')
    best_params = (0,0,0)
    n = 0
    n_failed = 0
    p0 = -60
    for p2 in range(-50, 50, 1):
        for p1 in range(-50, 1):
            n += 1
            try:
                def tau_func(t,q,qd):
                    return p2*t**2 + p1*t + p0
                distance = sim.simulate_with_controller(tau_func, **sim_args)
                if distance > best_distance:
                    best_distance = distance
                    best_params = (p2, p1, p0)
                    print("Ball was thrown {0} meters with {1}\
                          ".format(distance, best_params))
            except RuntimeError as e:
                n_failed += 1
                pass

    print(n, n_failed)



if __name__ == "__main__":
    main()
