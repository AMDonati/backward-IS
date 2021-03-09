import argparse
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.BackwardIS_smoother import RNNBackwardISSmoothing
from smc.utils import estimation_function_X0
import numpy as np
import os
import torch


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the observations and states")
    parser.add_argument("-model_path", type=str, required=True, help="path for uploading the rnn model path")
    parser.add_argument("-backward_samples", type=int, default=4,
                        help="number of backward samples for the backward IS smoother")
    parser.add_argument("-sigma_init", type=float, default=0.5,
                        help="covariance matrix for initial hidden state")
    parser.add_argument("-sigma_h", type=float, default=0.5,
                        help="covariance matrix for the internal gaussian noise for the transition function.")
    parser.add_argument("-sigma_y", type=float, default=0.5,
                        help="covariance matrix for the internal gaussian noise for the observation model.")
    return parser


def run(args):
    # upload observations and samples
    observations = np.load(os.path.join(args.data_path, "observations.npy"))
    states = np.load(os.path.join(args.data_path, "states.npy"))
    observations = torch.tensor(observations, requires_grad=False)
    states = torch.tensor(states, requires_grad=False)

    num_particles = observations.size(1)
    # upload trained rnn
    rnn = torch.load(args.model_path)
    rnn.update_sigmas(sigma_init=args.sigma_init, sigma_h=args.sigma_h, sigma_y=args.sigma_y)
    # Create the bootstrap filter
    rnn_bootstrap_filter = RNNBootstrapFilter(num_particles=num_particles, rnn=rnn)

    # Create the Backward IS Smoother
    backward_is_smoother = RNNBackwardISSmoothing(bootstrap_filter=rnn_bootstrap_filter, observations=observations,
                                                  states=states, backward_samples=args.backward_samples, estimation_function=estimation_function_X0)

    phi = backward_is_smoother.estimate_conditional_expectation_of_function()
    loss = backward_is_smoother.compute_mse_phi_X0(phi)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
