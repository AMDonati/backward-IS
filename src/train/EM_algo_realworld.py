import numpy as np
import torch
from smc.BootstrapFilter import SVBootstrapFilter
from smc.SV_smoother import SVBackwardISSmoothing, PoorManSmoothing
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os, json
import argparse
import time


def plot_EM_results(results, out_folder, out_file):
    plt.plot(np.linspace(0, len(results), len(results)), results)
    plt.savefig(os.path.join(out_folder, out_file))
    plt.close()

def plot_observations(observations, generated_observations, out_file):
    fig, ax = plt.subplots(figsize=(30, 15))
    x = np.linspace(0, len(generated_observations), len(generated_observations))
    ax.plot(x, observations, color='green', label='true observations')
    ax.plot(x, generated_observations, color='blue', label='generated observations')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.savefig(out_file)
    plt.close()


def plot_evol_params(list_params, true_params, out_file):
    alphas = [params[0] for params in list_params]
    true_alpha = true_params[0]
    sigmas = [params[1] for params in list_params]
    true_sigma = true_params[1]
    betas = [params[2] for params in list_params]
    true_beta = true_params[2]
    x = np.linspace(0, len(list_params), len(list_params))
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.plot(x, alphas, color='green')
    ax.plot(x, sigmas, color='blue')
    ax.plot(x, betas, color='red')
    ax.hlines(y=true_alpha, xmin=0, xmax=len(list_params), linestyles='dotted', color='green')
    ax.hlines(y=true_sigma, xmin=0, xmax=len(list_params), linestyles='dotted', color='blue')
    ax.hlines(y=true_beta, xmin=0, xmax=len(list_params), linestyles='dotted', color='red')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(out_file)
    plt.close()

def generate_observations_from_learned_params(results, seq_len):
    alpha = results.x[0]
    rho = results.x[1]
    sigma = np.exp(rho / 2)
    mu = results.x[2]
    beta = np.exp(mu/2)

    np.random.seed(123)# seed does not work...
    scale0 = alpha / np.sqrt(1 - alpha ** 2)
    X0 = np.random.normal(scale=scale0)
    print("initial state", X0)
    X = X0
    observations = np.zeros(seq_len)
    states = np.zeros(seq_len)

    print("generating data...")
    # generate synthetic dataset
    for k in range(seq_len):
        states[k] = X
        next_X = alpha * X + sigma * np.random.normal()
        Y = beta * np.exp(next_X / 2) * np.random.normal()
        observations[k] = Y
        X = next_X
    return observations


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str,
                        help="path for uploading the observations and states")
    parser.add_argument("-out_path", type=str, default="experiments_realworld")
    parser.add_argument("-results_path", type=str, default=None)
    parser.add_argument("-num_particles", type=int, default=100,
                        help="number of particles for the Bootstrap Filter")
    parser.add_argument("-backward_samples", type=int, default=16,
                        help="number of backward samples for the backward IS smoother")
    parser.add_argument("-algo", type=str, default="BIS",
                        help="PMS or BIS")
    parser.add_argument("-n_iter", type=int, default=50,
                        help="number of iterations for the EM algo.")
    parser.add_argument("-n_trials", type=int, default=5,
                        help="number of trials for state estimation.")
    parser.add_argument("-alpha", type=float, default=0.91,
                        help="init alpha for the EM algo.")
    parser.add_argument("-sigma", type=float, default=1.0,
                        help="init sigma for the EM algo.")
    parser.add_argument("-beta", type=float, default=0.5,
                        help="init beta for the EM algo.")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    #  ------------------------------------------------------------ hparams ------------------------------------------------#
    num_particles = args.num_particles
    backward_samples = args.backward_samples
    n_iter = args.n_iter
    algo = args.algo

    alpha = args.alpha
    sigma = args.sigma
    beta = args.beta

    init_params = [alpha, np.log(sigma**2), np.log(beta**2)]

    if args.results_path is None:
        # create out_folder for saving plots:
        out_folder = args.out_path
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
    else:
        out_folder = args.results_path

    # ----------------------------------------- create synthetic SV dataset --------------------------------------
    # values paper Jimmy: 0.8,O.1,1.
    # other params: 0.91, 1.0, 0.5

    observations_raw = np.load(os.path.join(args.data_path, "GE_observations.npy"))

    print("OBSERVATIONS", observations_raw)

    observations = observations_raw[np.newaxis, :]
    observations = np.repeat(observations, num_particles, axis=0)
    # observations = torch tensor of size T.
    observations = torch.tensor(observations, dtype=torch.float32)


    # ---------------------------------------------------------------------------------------------------------------------------- #

    if args.results_path is None:
        # Create bootstrap filter with init params and number of particles
        bt_filter = SVBootstrapFilter(num_particles, init_params)

        ######################### ---- Parameter Estimation with EM algo --------------- ###################################
        optim_method = 'Powell'  # optimizers tried: 'BFGS', 'Nelder-Mead', 'Powell', 'L-BFGS-B'
        # BFGS, L-BFGS -> params do not move.
        out_folder = os.path.join(out_folder,
                                  "{}_EM_{}_{}P_{}J-initparams-{}-{}-{}".format(algo, optim_method, num_particles,
                                                                                    backward_samples,
                                                                                    init_params[0],
                                                                                    init_params[1], init_params[2]))
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        maxiter = 300
        options = {'maxiter': maxiter, 'maxfev': None, 'disp': True, 'return_all': True}

        print("INIT PARAMS: {}".format(init_params))

        if algo == "BIS":
            smoother = SVBackwardISSmoothing(backward_samples=backward_samples, observations=observations,
                                             bootstrap_filter=bt_filter)
        elif algo == "PMS":
            smoother = PoorManSmoothing(bootstrap_filter=bt_filter, observations=observations)

        bt_filter.update_SV_params(init_params)

        # EM algo.
        expectations_results, list_params = [], [init_params]

        start_time_ = time.time()
        for iter in range(n_iter):
            # eval Q(\theta_k, \theta_k)

            print("PARAMS in SMOOTHING:", smoother.bootstrap_filter.params)
            # expectation = smoother.estimate_conditional_expectation_of_function(bt_filter.params)

            start_time = time.time()
            smoother.save_smoothing_elements()
            expectation = smoother.compute_expectation_from_saved_elements(params=bt_filter.params)
            expectations_results.append(expectation)
            print("eval Q(theta_k, theta_k) at iter {}: {}".format(iter, expectation))

            results = opt.minimize(fun=smoother.compute_expectation_from_saved_elements, x0=bt_filter.params,
                                   method=optim_method, options=options)

            print("time for one EM", time.time() - start_time)

            if "allvecs" in results.keys():
                fn_evals = [smoother.compute_expectation_from_saved_elements(vecs) for vecs in results["allvecs"]]
                out_file = "EM_{}_maxiter{}_{}-{}iters.png".format(optim_method, maxiter, iter, n_iter)
                plot_EM_results(fn_evals, out_folder, out_file)

            print("OPTIM SUCCESS:", results.success)

            print("new params: {}".format(results.x))
            list_params.append(results.x)

            # eval Q(theta_{k+1), \theta_k)
            new_expectation = smoother.compute_expectation_from_saved_elements(results.x)
            print("eval Q(theta_k+1, theta_k) at iter {}: {}".format(iter, new_expectation))

            bt_filter.update_SV_params(results.x)

            print("-" * 30)

        print("EXPECTATIONS:", expectations_results)
        plot_EM_results(expectations_results, out_folder, out_file="expectation_results_{}".format(optim_method))
        plot_evol_params(list_params=list_params, true_params=[alpha, np.log(sigma**2), np.log(beta**2)],
                         out_file=os.path.join(out_folder, "plot_evol_params.png"))

        params_star = dict(zip(["alpha", "sigma", "beta"], [str(i) for i in results.x]))
        with open(os.path.join(out_folder, "params_star.json"), 'w') as f:
            json.dump(params_star, f)
        np.save(os.path.join(out_folder, "list_params.npy"), np.array(list_params))

        print("------------TIME FOR THE EM algo-----------")
        print(time.time()-start_time_)

        print("done with EM algo")

        generated_observations = generate_observations_from_learned_params(results, seq_len=observations.shape[-1])

        plot_observations(observations_raw, generated_observations, os.path.join(out_folder, 'plot_observations.png'))

        mse = 0.5 * np.square(observations_raw - generated_observations)
        np.save(os.path.join(out_folder, "mses.npy"), mse)
        np.save(os.path.join(out_folder, "generated_observations.npy"), generated_observations)

    else:
        # upload converged params:
        params_path = os.path.join(out_folder, "params_star.json")
        with open(params_path, 'r') as f:
            converged_params = json.load(f)
        alpha = float(converged_params["alpha"])
        sigma = float(converged_params["sigma"])
        beta = float(converged_params["beta"])
        bt_filter = SVBootstrapFilter(num_particles, [alpha, sigma, beta])

        index_states = [1, 24, 49, 99]
        num_trials = args.n_trials
        states_estims = np.zeros((len(index_states), num_trials))
        results = dict.fromkeys(index_states)
        for iter, index_state in enumerate(index_states):
            # create SVBackward IS smoothing with number of backward samples, out_folder, logger.
            if algo == 'BIS':
                smoother = SVBackwardISSmoothing(backward_samples=backward_samples, observations=observations,
                                                 bootstrap_filter=bt_filter, index_state=index_state)
            elif algo == 'PMS':
                smoother = PoorManSmoothing(observations=observations,
                                            bootstrap_filter=bt_filter, index_state=index_state)

            print("params of SV model:", smoother.bootstrap_filter.params)

            for trials in range(num_trials):
                start_time = time.time()
                if algo == "BIS":
                    state_estim = -smoother.estimate_conditional_expectation_of_function(params=[alpha, sigma, beta])
                elif algo == "PMS":
                    smoother.save_smoothing_elements()
                    state_estim = -smoother.compute_expectation_from_saved_elements([alpha, sigma, beta])
                print("time for one estimation", time.time() - start_time)
                states_estims[iter, trials] = state_estim

        np.save(os.path.join(out_folder, "state_estims.npy"), state_estim)



# Loop over number of iterations for the EM algo.
# estimate the E-step with backwardIS smoothing (get an estimation of the log likelihood)

# Max-step: argmax over  parameters
# update bootstrap filter params.

# class SVBackwardISSmoothing(SmoothingAlgo):
#     def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function, out_folder,
#                  index_state=0, save_elements=False, logger=None)
