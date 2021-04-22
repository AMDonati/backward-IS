import argparse
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.BackwardIS_smoother import RNNBackwardISSmoothing, PoorManSmoothing
from smc.utils import estimation_function_X
from train.utils import write_to_csv
import numpy as np
import os
import torch
from smc.plots import plot_variance_error, plot_variance_square_error, plot_mean_square_error

def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the observations and states")
    parser.add_argument("-model_path", type=str, required=True, help="path for uploading the rnn model path")
    parser.add_argument("-out_path", type=str, default="experiments")
    parser.add_argument("-num_particles", type=int, default=1000,
                        help="number of particles for the Bootstrap Filter")
    parser.add_argument("-particles_pms", type=int, default=1000,
                        help="number of particles for the Bootstrap Filter for PMS algo.")
    parser.add_argument("-backward_samples", type=int, default=32,
                        help="number of backward samples for the backward IS smoother")
    parser.add_argument("-sigma_init", type=float, default=0.1,
                        help="covariance matrix for initial hidden state")
    parser.add_argument("-sigma_h", type=float, default=0.1,
                        help="covariance matrix for the internal gaussian noise for the transition function.")
    parser.add_argument("-sigma_y", type=float, default=0.1,
                        help="covariance matrix for the internal gaussian noise for the observation model.")
    parser.add_argument("-debug", type=int, default=1,
                        help="debug smoothing algo or not.")
    parser.add_argument("-index_states", nargs='+', type=int, default=[0],
                        help='index of states to estimate.')
    parser.add_argument("-backward_is", type=int, default=1,
                        help="debug smoothing algo or not.")
    parser.add_argument("-pms", type=int, default=1,
                        help="debug smoothing algo or not.")
    return parser

def fill_dict_pms(dict, index_state, results_pms, stack=False):
    dict[index_state]["pms_all_seq"] = results_pms[-1]
    if stack:
        dict[index_state]["pms_by_seq"] = torch.stack(results_pms, dim=0).squeeze(1).cpu().numpy()
    else:
        dict[index_state]["pms_by_seq"] = results_pms
    return dict

def fill_dict_backward(dict, index_state, results_backward, stack=False):
    dict[index_state]["backward_all_seq"] = results_backward[-1]
    if stack:
        dict[index_state]["backward_by_seq"] = torch.stack(results_backward, dim=0).squeeze(1).cpu().numpy()
    else:
        dict[index_state]["backward_by_seq"] = results_backward
    return dict

def run(args):
    # upload observations and samples
    observations = np.load(os.path.join(args.data_path, "observations.npy"))
    states = np.load(os.path.join(args.data_path, "states.npy"))
    observations = torch.tensor(observations, requires_grad=False, dtype=torch.float32)
    states = torch.tensor(states, requires_grad=False, dtype=torch.float32)

    # create out_folder:
    out_path = os.path.join(args.data_path, args.out_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    backward_is_out = os.path.join(args.data_path, args.out_path, "backward_is")
    if not os.path.isdir(backward_is_out):
        os.makedirs(backward_is_out)
    pms_out = os.path.join(args.data_path, args.out_path, "pms")
    if not os.path.isdir(pms_out):
        os.makedirs(pms_out)

    # upload trained rnn
    rnn = torch.load(args.model_path)
    rnn.update_sigmas(sigma_init=args.sigma_init, sigma_h=args.sigma_h, sigma_y=args.sigma_y)
    # Create the bootstrap filter
    rnn_bootstrap_filter = RNNBootstrapFilter(num_particles=args.num_particles, rnn=rnn)
    rnn_bootstrap_filter_pms = RNNBootstrapFilter(num_particles=args.particles_pms, rnn=rnn)

    # Estimate $mathbb[E][X_k | Y_{0:n}]$ for several values of k
    index_states = args.index_states
    dict_results = dict.fromkeys(index_states)
    for key in dict_results.keys():
        dict_results[key] = {"pms_all_seq": [], "backward_all_seq": [], "pms_by_seq":[], "backward_by_seq":[]}
    dict_errors = dict.fromkeys(index_states)
    for key in dict_errors.keys():
        dict_errors[key] = {"pms_all_seq": [], "backward_all_seq": [], "pms_by_seq":[], "backward_by_seq":[]}
    particles_backward, weights_backward = [], []

    if args.backward_is:

        for index_state in index_states:
            # Create the Backward IS Smoother
            results_backward, errors_backward, phis_backward = [], [], []
            backward_is_smoother = RNNBackwardISSmoothing(bootstrap_filter=rnn_bootstrap_filter,
                                                          observations=observations,
                                                          states=states, backward_samples=args.backward_samples,
                                                          estimation_function=estimation_function_X,
                                                          save_elements=args.debug, index_state=index_state,
                                                          out_folder=out_path)
            backward_is_smoother.logger.info(
                "-------------------------------------------------------------- ESTIMATING X_{}-----------------------------------------------------------------------------".format(
                    index_state))

            # compute backward IS smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
            backward_is_smoother.logger.info(
                "--------------------------------------------BACKWARD IS--------------------------------------------------------")
            (mses_backward_is, errors_backward_is), phi_backward_is, (
            particle_backward, weight_backward) = backward_is_smoother.estimate_conditional_expectation_of_function()
            print("Loss backward IS smoother", mses_backward_is[-1])
            np.save(os.path.join(backward_is_out, "phis_backward_X_{}.npy".format(index_state)),
                    phi_backward_is.cpu().squeeze().numpy())

            particles_backward.append(particle_backward)
            weights_backward.append(weight_backward)
            backward_is_smoother.logger.info(
            "--------------------------------------------------------------------------------------------------------------------------------------------")

            dict_results = fill_dict_backward(dict=dict_results, index_state=index_state, results_backward=mses_backward_is)
            dict_errors = fill_dict_backward(dict=dict_errors, index_state=index_state, results_backward=errors_backward_is, stack=True)

        poor_man_smoother = PoorManSmoothing(bootstrap_filter=rnn_bootstrap_filter_pms,
                                         observations=observations,
                                         states=states,
                                         estimation_function=estimation_function_X,
                                         index_state=index_state,
                                         out_folder=out_path, logger=backward_is_smoother.logger)

    else:
        poor_man_smoother = PoorManSmoothing(bootstrap_filter=rnn_bootstrap_filter_pms,
                                             observations=observations,
                                             states=states,
                                             estimation_function=estimation_function_X,
                                             index_state=args.index_states[0],
                                             out_folder=out_path)

    if args.pms:
        # compute poor man smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
        poor_man_smoother.logger.info(
            "--------------------------------------------POOR MAN --------------------------------------------------------")
        (mses_pms, errors_pms), phi_pms, _ = poor_man_smoother.estimate_conditional_expectation_of_function()
        print("loss PMS smoother", mses_pms[-1])
        poor_man_smoother.logger.info(
            "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        #np.save(os.path.join(pms_out, "phis_pms.npy"), phis_pms[index_states].cpu().squeeze().numpy())
        for index in index_states:
            res_pms = [r[index] for r in mses_pms]
            err_pms = [e[index] for e in errors_pms]
            dict_results = fill_dict_pms(dict=dict_results, index_state=index, results_pms=res_pms)
            dict_errors = fill_dict_pms(dict=dict_errors, index_state=index, results_pms=err_pms,
                                             stack=True)

        particles_pms = poor_man_smoother.trajectories[:,:,index_states,:]

    if args.backward_is:
        seq_errors_backward = [dict_errors[k]["backward_by_seq"] for k in dict_errors.keys()]
        seq_loss_backward = [dict_results[k]["backward_by_seq"] for k in dict_results.keys()]
        for k in dict_errors.keys():
            np.save(os.path.join(backward_is_out, "backward_errors_X_{}".format(k)),
                    dict_errors[k]["backward_by_seq"].squeeze())
    else:
        seq_errors_backward, seq_loss_backward = [], []
    if args.pms:
        seq_loss_pms = [dict_results[k]["pms_by_seq"] for k in dict_results.keys()]
        seq_errors_pms = [dict_errors[k]["pms_by_seq"] for k in dict_errors.keys()]
        for k in dict_errors.keys():
            np.save(os.path.join(pms_out, "pms_errors_X_{}".format(k)), dict_errors[k]["pms_by_seq"].squeeze())
    else:
        seq_errors_pms, seq_loss_pms = [], []

    if args.backward_is and args.pms:
        backward_is_smoother.boxplots_error(errors_backward=seq_errors_backward, errors_pms=seq_errors_pms,
                                            out_folder=backward_is_out, num_runs=args.runs, index_states=index_states)
        backward_is_smoother.boxplots_loss(loss_backward=seq_loss_backward, loss_pms=seq_loss_pms,
                                            out_folder=backward_is_out, num_runs=args.runs, index_states=index_states)
        backward_is_smoother.plot_particles_all_k(particles_backward=particles_backward, weights_backward=weights_backward,
                                                  particles_pms=particles_pms, weights_pms=poor_man_smoother.filtering_weights,
                                                  out_folder=backward_is_out, num_runs=args.runs, index_states=index_states)


    plot_mean_square_error(dict_results=dict_results, num_runs=args.runs, out_folder=backward_is_out, num_particles=args.num_particles, backward_samples=args.backward_samples, args=args)
    plot_variance_square_error(dict_results=dict_results, num_runs=args.runs, out_folder=backward_is_out,
                           num_particles=args.num_particles, backward_samples=args.backward_samples, args=args)
    plot_variance_error(dict_errors=dict_errors, num_runs=args.runs, out_folder=backward_is_out, num_particles=args.num_particles, backward_samples=args.backward_samples, args=args)
    write_to_csv(output_dir=os.path.join(out_path, "results_{}runs_{}J_{}particles_{}pms-part.csv".format(args.runs, args.backward_samples, args.num_particles, args.particles_pms)), dic=dict_results)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
