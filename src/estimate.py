import argparse
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.BackwardIS_smoother import RNNBackwardISSmoothing, PoorManSmoothing
from smc.utils import estimation_function_X
from train.utils import write_to_csv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str, required=True, help="path for uploading the observations and states")
    parser.add_argument("-model_path", type=str, required=True, help="path for uploading the rnn model path")
    parser.add_argument("-out_path", type=str, default="experiments")
    parser.add_argument("-num_particles", type=int, default=10,
                        help="number of particles for the Bootstrap Filter")
    parser.add_argument("-particles_pms", type=int, default=5,
                        help="number of particles for the Bootstrap Filter for PMS algo.")
    parser.add_argument("-backward_samples", type=int, default=4,
                        help="number of backward samples for the backward IS smoother")
    parser.add_argument("-sigma_init", type=float, default=0.1,
                        help="covariance matrix for initial hidden state")
    parser.add_argument("-sigma_h", type=float, default=0.1,
                        help="covariance matrix for the internal gaussian noise for the transition function.")
    parser.add_argument("-sigma_y", type=float, default=0.1,
                        help="covariance matrix for the internal gaussian noise for the observation model.")
    parser.add_argument("-debug", type=int, default=1,
                        help="debug smoothing algo or not.")
    parser.add_argument("-index_states", nargs='+', type=int, default=[0,10],
                        help='index of states to estimate.')
    parser.add_argument("-runs", type=int, default=1,
                        help="number of runs for the smoothing algo.")
    parser.add_argument("-backward_is", type=int, default=1,
                        help="debug smoothing algo or not.")
    parser.add_argument("-pms", type=int, default=0,
                        help="debug smoothing algo or not.")
    return parser

def fill_dict_pms(dict, index_state, results_pms, stack=False, stats=True):
    if stats:
        dict[index_state]["pms_mean"] = np.round(np.mean(results_pms), 4)
        dict[index_state]["pms_var"] = np.round(np.var(results_pms), 8)
    if stack:
        dict[index_state]["pms_runs"] = torch.stack(results_pms, dim=0).squeeze(1).cpu().numpy()
    else:
        dict[index_state]["pms_runs"] = results_pms
    return dict

def fill_dict_backward(dict, index_state, results_backward, stack=False, stats=True):
    if stats:
        dict[index_state]["backward_is_mean"] = np.round(np.mean(results_backward), 4)
        dict[index_state]["backward_is_var"] = np.round(np.var(results_backward), 8)
    if stack:
        dict[index_state]["backward_runs"] = torch.stack(results_backward, dim=0).squeeze(1).cpu().numpy()
    else:
        dict[index_state]["backward_runs"] = results_backward
    return dict

def plot_mean_square_error(dict_results, num_runs, out_folder, num_particles, backward_samples, args):
    if args.backward_is:
        backward_is_mean = [dict_results[k]["backward_is_mean"] for k in dict_results.keys()]
    if args.pms:
        pms_mean = [dict_results[k]["pms_mean"] for k in dict_results.keys()]
    fig, ax = plt.subplots(figsize=(25, 10))
    if args.pms:
        xx = np.linspace(1, len(pms_mean), len(pms_mean))
    if args.backward_is:
        xx = np.linspace(1, len(backward_is_mean), len(backward_is_mean))
    if args.backward_is:
        ax.plot(xx, backward_is_mean, color='blue', marker='x', label='backward IS Smoother')
    if args.pms:
        ax.plot(xx, pms_mean, color='red', label='PMS Smoother')
    labels = ['X_{}'.format(k) for k in list(dict_results.keys())]
    plt.xticks(ticks=xx, labels=labels)
    ax.grid('on')
    ax.legend(loc='upper center', fontsize=16)
    ax.set_title('mean squared error', fontsize=20)
    out_file = "mse_{}runs_{}particles_{}J".format(num_runs, num_particles, backward_samples)
    fig.savefig(os.path.join(out_folder, out_file))
    plt.close()

def plot_variance_error(dict_errors, num_runs, out_folder, num_particles, backward_samples, args):
    if args.backward_is:
        backward_is_var_1 = [np.var(dict_errors[k]["backward_runs"][:,0]) for k in dict_errors.keys()]
        backward_is_var_2 = [np.var(dict_errors[k]["backward_runs"][:,1]) for k in dict_errors.keys()]
    if args.pms:
        pms_var_1 = [np.var(dict_errors[k]["pms_runs"][:, 0]) for k in dict_errors.keys()]
        pms_var_2 = [np.var(dict_errors[k]["pms_runs"][:, 1]) for k in dict_errors.keys()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
    if args.pms:
        xx = np.linspace(1, len(pms_var_1), len(pms_var_1))
    elif args.backward_is:
        xx = np.linspace(1, len(backward_is_var_1), len(backward_is_var_1))
    labels = ['X_{}'.format(k) for k in list(dict_errors.keys())]
    plt.sca(ax1)
    plt.xticks(ticks=xx, labels=labels)
    if args.backward_is:
        ax1.plot(xx, backward_is_var_1, color='blue', marker='x', label='Backward IS Smoother')
    if args.pms:
        ax1.plot(xx, pms_var_1, color='red', marker='x', label='PMS Smoother')
    plt.sca(ax2)
    plt.xticks(ticks=xx, labels=labels)
    if args.backward_is:
        ax2.plot(xx, backward_is_var_2, color='blue', marker='x')
    if args.pms:
        ax2.plot(xx, pms_var_2, color='red', marker='x')
    ax1.grid('on')
    ax2.grid('on')
    ax1.legend('upper center', fontsize=16)
    ax1.set_title('variance of the estimation error', fontsize=20)
    out_file = "error_variances_{}runs_{}particles_{}J".format(num_runs, num_particles, backward_samples)
    fig.savefig(os.path.join(out_folder, out_file))
    plt.close()

def plot_variance_square_error(dict_results, num_runs, out_folder, num_particles, backward_samples, args):
    if args.backward_is:
        backward_is_mean = [dict_results[k]["backward_is_var"] for k in dict_results.keys()]
    if args.pms:
        pms_mean = [dict_results[k]["pms_var"] for k in dict_results.keys()]
    fig, ax = plt.subplots(figsize=(25, 10))
    len_ = len(pms_mean) if args.pms else len(backward_is_mean)
    xx = np.linspace(1, len_, len_)
    if args.backward_is:
        ax.plot(xx, backward_is_mean, color='blue', marker='x', label='backward IS smoother')
    if args.pms:
        ax.plot(xx, pms_mean, color='red', marker='x', label='PMS smoother')
    labels = ['X_{}'.format(k) for k in list(dict_results.keys())]
    plt.xticks(ticks=xx, labels=labels)
    ax.legend('upper center', fontsize=16)
    out_file = "square_error_var_{}runs_{}particles_{}J".format(num_runs, num_particles, backward_samples)
    ax.grid('on')
    ax.set_title('variance of the squared error', fontsize=20)
    fig.savefig(os.path.join(out_folder, out_file))
    plt.close()

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
        dict_results[key] = {"pms_mean": [], "pms_var": [], "backward_is_mean": [], "backward_is_var": [], "pms_runs":[], "backward_runs":[]}
    dict_errors = dict.fromkeys(index_states)
    for key in dict_errors.keys():
        dict_errors[key] = {"pms_runs":[], "backward_runs":[]}
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

            for _ in range(args.runs):
                # compute backward IS smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
                backward_is_smoother.logger.info("--------------------------------------------BACKWARD IS--------------------------------------------------------")
                phi_backward_is, (particle_backward, weight_backward) = backward_is_smoother.estimate_conditional_expectation_of_function()
                loss_backward_is, error_backward_is = backward_is_smoother.compute_mse_phi_X0(phi_backward_is)
                print("Loss backward IS smoother", loss_backward_is)
                results_backward.append(round(loss_backward_is.item(), 4))
                phis_backward.append(phi_backward_is)
                errors_backward.append(error_backward_is)
            particles_backward.append(particle_backward)
            weights_backward.append(weight_backward)
            backward_is_smoother.logger.info(
            "--------------------------------------------------------------------------------------------------------------------------------------------")

            dict_results = fill_dict_backward(dict=dict_results, index_state=index_state, results_backward=results_backward)
            dict_errors = fill_dict_backward(dict=dict_errors, index_state=index_state, results_backward=errors_backward, stack=True, stats=False)

        np.save(os.path.join(backward_is_out, "phis_backward.npy"), torch.stack(phis_backward).cpu().squeeze().numpy())
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

    results_pms, errors_pms, phis_pms = [], [], []

    if args.pms:

        for _ in range(args.runs):
            # compute poor man smoothing estimation of $mathbb[E][X_0|Y_{0:n}]$
            poor_man_smoother.logger.info(
            "--------------------------------------------POOR MAN --------------------------------------------------------")
            indices_matrix, particles_seq = poor_man_smoother.estimate_conditional_expectation_of_function()
            error_pms, loss_pms, phi_pms = poor_man_smoother.get_error()
            print("loss PMS smoother", loss_pms)
            results_pms.append(loss_pms)
            errors_pms.append(error_pms)
            phis_pms.append(phi_pms[index_states])
            poor_man_smoother.logger.info(
            "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        np.save(os.path.join(pms_out, "phis_pms.npy"), torch.stack(phis_pms).cpu().squeeze().numpy())
        for index in index_states:
            res_pms = [r[index] for r in results_pms]
            err_pms = [e[index] for e in errors_pms]
            dict_results = fill_dict_pms(dict=dict_results, index_state=index, results_pms=res_pms)
            dict_errors = fill_dict_pms(dict=dict_errors, index_state=index, results_pms=err_pms,
                                             stack=True, stats=False)

        particles_pms = poor_man_smoother.trajectories[:,:,index_states,:]

    if args.backward_is:
        seq_errors_backward = [dict_errors[k]["backward_runs"] for k in dict_errors.keys()]
        seq_loss_backward = [dict_results[k]["backward_runs"] for k in dict_results.keys()]
        for k in dict_errors.keys():
            np.save(os.path.join(backward_is_out, "backward_errors_X_{}".format(k)),
                    dict_errors[k]["backward_runs"].squeeze())
    else:
        seq_errors_backward, seq_loss_backward = [], []
    if args.pms:
        seq_loss_pms = [dict_results[k]["pms_runs"] for k in dict_results.keys()]
        seq_errors_pms = [dict_errors[k]["pms_runs"] for k in dict_errors.keys()]
        for k in dict_errors.keys():
            np.save(os.path.join(pms_out, "pms_errors_X_{}".format(k)), dict_errors[k]["pms_runs"].squeeze())
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
