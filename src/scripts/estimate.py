import argparse
from smc.BootstrapFilter import BootstrapFilter
from smc.BackwardIS_smoothing import RNNBackwardISSmoothing
import numpy as np
import os

def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the observations and states")
    return parser

def run(args):
    # upload observations and samples
    observations = np.load(os.path.join(args.data_path, "observations.npy"))
    states = np.load(os.path.join(args.data_path, "states.npy"))

    if args.ep > 0:
        algo.train()
    else:
        print("skipping training...")
    algo.test()
    algo.generate_observations(sigma_init=args.sigma_init, sigma_h=args.sigma_h, sigma_y=args.sigma_y, num_samples=args.num_samples)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)



