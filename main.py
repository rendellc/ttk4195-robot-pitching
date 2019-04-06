import simulation as sim
import numpy as np

def main():
    """ Main function for module """
    sim_args = {}
    sim_args['speed_factor'] = 1.0
    sim_args['step_size'] = 0.01
    #sim_args['print_info'] = False

    tau = lambda t: 0 + 30*t
    distance = sim.simulate_with_controller(tau, **sim_args)
    print("Ball was thrown {0} meters".format(distance))


if __name__ == "__main__":
    main()
