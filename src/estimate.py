import argparse
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.BackwardIS_smoother import RNNBackwardISSmoothing, PoorManSmoothing
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
    parser.add_argument("-index_states", nargs='+', type=int, default=list(range(24)),
                        help='index of states to estimate.')
    parser.add_argument("-runs", type=int, default=1,
                        help="number of runs for the smoothing algo.")
    return parser

def fill_dict(dict, index_state, results_backward, results_pms, stack=False, stats=True):
    if stats:
        dict[index_state]["backward_is_mean"] = np.round(np.mean(results_backward), 4)
        dict[index_state]["backward_is_var"] = np.round(np.var(results_backward), 8)
        dict[index_state]["pms_mean"] = np.round(np.mean(results_pms), 4)
        dict[index_state]["pms_var"] = np.round(np.var(results_pms), 8)
    if stack:
        dict[index_state]["pms_runs"] = torch.stack(results_pms, dim=0).squeeze(1).cpu().numpy()
        dict[index_state]["backward_runs"] = torch.stack(results_backward, dim=0).squeeze(1).cpu().numpy()
    else:
        dict[index_state]["pms_runs"] = results_pms
        dict[index_state]["backward_runs"] = results_backward
    return dict

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
    index_states = args.index_states
    dict_results = dict.fromkeys(index_states)
    for key in dict_results.keys():
        dict_results[key] = {"pms_mean": [], "pms_var": [], "backward_is_mean": [], "backward_is_var": [], "pms_runs":[], "backward_runs":[]}
    dict_errors = dict.fromkeys(index_states)
    for key in dict_errors.keys():
        dict_errors[key] = {"pms_runs":[], "backward_runs":[]}
    particles_backward, weights_backward = [], []
    particles_pms, weights_pms = [], []
    trajectories_pms = []
    for index_state in index_states:
        # Create the Backward IS Smoother
        results_backward, results_pms = [], []
        phis_pms, phis_backward = [], []
        errors_backward, errors_pms = [], []
        backward_is_smoother = RNNBackwardISSmoothing(bootstrap_filter=rnn_bootstrap_filter,
                                                      observations=observations,
                                                      states=states, backward_samples=args.backward_samples,
                                                      estimation_function=estimation_function_X,
                                                      save_elements=args.debug, index_state=index_state,
                                                      out_folder=args.data_path)
        backward_is_smoother.logger.info(
            "-------------------------------------------------------------- ESTIMATING X_{}-----------------------------------------------------------------------------".format(
                index_state))
        poor_man_smoother = PoorManSmoothing(bootstrap_filter=rnn_bootstrap_filter,
                                             observations=observations,
                                             states=states,
                                             estimation_function=estimation_function_X,
                                             index_state=index_state,
                                             out_folder=args.data_path)

        for _ in range(args.runs):
            # compute backward IS smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
            backward_is_smoother.logger.info("--------------------------------------------BACKWARD IS--------------------------------------------------------")
            phi_backward_is, (particle_backward, weight_backward) = backward_is_smoother.estimate_conditional_expectation_of_function()
            #backward_is_smoother.debug_elements(data_path=backward_is_out)
            loss_backward_is, error_backward_is = backward_is_smoother.compute_mse_phi_X0(phi_backward_is)
            #backward_is_smoother.plot_estimation_versus_state(phi=phi_backward_is, out_folder=backward_is_out)
            print("Loss backward IS smoother", loss_backward_is)
            results_backward.append(round(loss_backward_is.item(), 4))
            phis_backward.append(phi_backward_is)
            errors_backward.append(error_backward_is)
            particles_backward.append(particle_backward)
            weights_backward.append(weight_backward)
            backward_is_smoother.logger.info(
                "--------------------------------------------------------------------------------------------------------------------------------------------")

            # compute poor man smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
            backward_is_smoother.logger.info(
                "--------------------------------------------POOR MAN --------------------------------------------------------")
            phi_pms, (particle_pms, weight_pms, trajectory_pms), (indices_matrix, particles_seq) = poor_man_smoother.estimate_conditional_expectation_of_function()
            loss_pms, error_pms = poor_man_smoother.compute_mse_phi_X0(phi_pms)

            error_pms_bis, loss_pms_bis = poor_man_smoother.get_error()
            #poor_man_smoother.plot_estimation_versus_state(phi=phi_pms, out_folder=pms_out)
            print("Loss poor man smoother", loss_pms)
            print("checking errors", error_pms_bis[index_state] - error_pms)
            print("checking loss", loss_pms_bis[index_state] - loss_pms)
            results_pms.append(round(loss_pms.item(), 4))
            phis_pms.append(phi_pms)
            errors_pms.append(error_pms)
            particles_pms.append(particle_pms)
            weights_pms.append(weight_pms)
            trajectories_pms.append(trajectory_pms)
            backward_is_smoother.logger.info(
                "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

            genealogy = poor_man_smoother.get_genealogy(indices_matrix=indices_matrix)
            resampled_trajectories = poor_man_smoother.resample_trajectories(genealogy=genealogy, trajectories=particles_seq)

            backward_is_smoother.logger.info(
                "-------------------------------- PMS  PARTICLES GENEALOGY--------------------------------")
            backward_is_smoother.logger.info(genealogy)

        #backward_is_smoother.plot_multiple_runs(phis_backward=phis_backward, phis_pms=phis_pms,
                                                #out_folder=backward_is_out, num_runs=args.runs)

        dict_results = fill_dict(dict=dict_results, index_state=index_state, results_backward=results_backward, results_pms=results_pms)
        dict_errors = fill_dict(dict=dict_errors, index_state=index_state, results_backward=errors_backward, results_pms=errors_pms, stack=True, stats=False)

    seq_errors_backward = [dict_errors[k]["backward_runs"] for k in dict_errors.keys()]
    seq_errors_pms = [dict_errors[k]["pms_runs"] for k in dict_errors.keys()]
    seq_loss_backward = [dict_results[k]["backward_runs"] for k in dict_results.keys()]
    seq_loss_pms = [dict_results[k]["pms_runs"] for k in dict_results.keys()]
    backward_is_smoother.boxplots_error(errors_backward=seq_errors_backward, errors_pms=seq_errors_pms,
                                        out_folder=backward_is_out, num_runs=args.runs)
    backward_is_smoother.boxplots_loss(loss_backward=seq_loss_backward, loss_pms=seq_loss_pms,
                                        out_folder=backward_is_out, num_runs=args.runs)

    backward_is_smoother.plot_trajectories_pms(trajectories=trajectory_pms, out_folder=pms_out)

    if args.runs == 1:
        backward_is_smoother.plot_particles_all_k(particles_backward=particles_backward, weights_backward=weights_backward, particles_pms=particles_pms, weights_pms=weights_pms, out_folder=backward_is_out, num_runs=args.runs)

    write_to_csv(output_dir=os.path.join(args.data_path, "results_{}.csv".format(args.runs)), dic=dict_results)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
