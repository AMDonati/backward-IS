import argparse
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.BackwardIS_smoother import RNNBackwardISSmoothing
from smc.utils import estimation_function_X
from train.utils import write_to_csv
import numpy as np
import os
import torch


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the observations and states")
    parser.add_argument("-model_path", type=str, required=True, help="path for uploading the rnn model path")
    parser.add_argument("-num_particles", type=int, default=100,
                        help="number of particles for the Bootstrap Filter")
    parser.add_argument("-backward_samples", type=int, default=4,
                        help="number of backward samples for the backward IS smoother")
    parser.add_argument("-sigma_init", type=float, default=0.5,
                        help="covariance matrix for initial hidden state")
    parser.add_argument("-sigma_h", type=float, default=0.5,
                        help="covariance matrix for the internal gaussian noise for the transition function.")
    parser.add_argument("-sigma_y", type=float, default=0.5,
                        help="covariance matrix for the internal gaussian noise for the observation model.")
    parser.add_argument("-debug", type=int, default=1,
                        help="debug smoothing algo or not.")
    return parser


def run(args):
    # upload observations and samples
    observations = np.load(os.path.join(args.data_path, "observations.npy"))
    states = np.load(os.path.join(args.data_path, "states.npy"))
    observations = torch.tensor(observations, requires_grad=False)
    states = torch.tensor(states, requires_grad=False)

    # create out_folder:
    backward_is_out = os.path.join(args.data_path, "backward_is")
    if not os.path.isdir(backward_is_out):
        os.makedirs(backward_is_out)
    pms_out = os.path.join(args.data_path, "pms")
    if not os.path.isdir(pms_out):
        os.makedirs(pms_out)

    # upload trained rnn
    rnn = torch.load(args.model_path)
    rnn.update_sigmas(sigma_init=args.sigma_init, sigma_h=args.sigma_h, sigma_y=args.sigma_y)
    # Create the bootstrap filter
    rnn_bootstrap_filter = RNNBootstrapFilter(num_particles=args.num_particles, rnn=rnn)

    # Estimate $mathbb[E][X_k | Y_{0:n}]$ for several values of k
    index_states = [0,1,5,11,17,23]
    dict_stats = dict.fromkeys(index_states)
    for key in dict_stats.keys():
        dict_stats[key] = {"pms": [], "backward_is": []}

    for index_state in index_states:
        print("-------------------------------------------------------------- k: {}-----------------------------------------------------------------------------".format(index_state))
        # Create the Backward IS Smoother
        backward_is_smoother = RNNBackwardISSmoothing(bootstrap_filter=rnn_bootstrap_filter, observations=observations,
                                                      states=states, backward_samples=args.backward_samples, estimation_function=estimation_function_X, save_elements=args.debug, index_state=index_state)
        # compute backward IS smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
        phi_backward_is = backward_is_smoother.estimate_conditional_expectation_of_function()
        backward_is_smoother.debug_elements(data_path=backward_is_out)
        loss_backward_is, var_backward_is = backward_is_smoother.compute_mse_phi_X0(phi_backward_is)
        backward_is_smoother.plot_estimation_versus_state(phi=phi_backward_is, out_folder=backward_is_out)
        print("Loss backward IS smoother", loss_backward_is)
        if observations.size(0) > 1:
            print("Variance of backward is loss over multiple observations", var_backward_is)
        dict_stats[index_state]["backward_is"] = round(loss_backward_is.item(), 4)

        # compute poor man smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
        phi_pms = backward_is_smoother.poor_man_smoother_estimation()
        loss_pms, var_pms = backward_is_smoother.compute_mse_phi_X0(phi_pms)
        backward_is_smoother.plot_estimation_versus_state(phi=phi_pms, out_folder=pms_out)
        print("Loss poor man smoother", loss_pms)
        if observations.size(0) > 1:
            print("Variance of poor man smoother loss over multiple observations", var_pms)
        dict_stats[index_state]["pms"] = round(loss_pms.item(), 4)
        print("-------------------------------------------------------------------------------------------------------------------------------------------")

    write_to_csv(output_dir=os.path.join(args.data_path, "results.csv"), dic=dict_stats)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
