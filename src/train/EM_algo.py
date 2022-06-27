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


def mean_expectation(params):
    expectations = np.zeros(n_bis)
    for i in range(n_bis):
        smoother.save_smoothing_elements()
        expectation = smoother.compute_expectation_from_saved_elements(params)
        expectations[i] = expectation
    return np.mean(expectations)


def plot_state_estimation(state_estims, true_states, index_states, out_file):
    fig, ax = plt.subplots()
    x = np.linspace(0, len(true_states), len(true_states))
    xx = np.linspace(0.25, len(true_states)+0.25, len(true_states))
    ax.scatter(xx, true_states, color='green', marker='x')
    for i in range(state_estims.shape[1]):
        ax.scatter(x, state_estims[:, i], color='blue')
    ax.set_xlabel(['X{}'.format(index) for index in index_states])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
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


def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-data_path", type=str,
                        help="path for uploading the observations and states")
    parser.add_argument("-out_path", type=str, default="experiments")
    parser.add_argument("-num_particles", type=int, default=100,
                        help="number of particles for the Bootstrap Filter")
    parser.add_argument("-backward_samples", type=int, default=16,
                        help="number of backward samples for the backward IS smoother")
    parser.add_argument("-init_params", type=str, default="random1",
                        help='initial params for the SV model.')
    parser.add_argument("-algo", type=str, default="BIS",
                        help="PMS or BIS")
    parser.add_argument("-n_iter", type=int, default=50,
                        help="number of iterations for the EM algo.")
    parser.add_argument("-n_trials", type=int, default=50,
                        help="number of trials for state estimation.")
    parser.add_argument("-seq_len", type=int, default=100,
                        help="number of observations.")
    parser.add_argument("-estim", type=str, default="parameter",
                        help="state or parameter estimation")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    #  ------------------------------------------------------------ hparams ------------------------------------------------#
    num_particles = args.num_particles
    backward_samples = args.backward_samples

    alpha = 0.98
    sigma = 0.1
    rho = np.log(sigma**2)
    beta = 0.05
    mu = np.log(beta**2)

    if args.init_params == "random1":
        init_params = [0.5, np.log(0.2**2), np.log(1.2**2)]
    if args.init_params == "random2":
        init_params = [0.75, np.log(1.2**2), np.log(0.3**2)]
    elif args.init_params == "true":
        init_params = [alpha-0.05, np.log((sigma+0.05)**2), np.log((beta+0.02)**2)]

    #init_params = [torch.tensor(i) for i in init_params]

    n_iter = args.n_iter
    n_bis = 1

    # state estimation or parameter estimation:
    if args.estim == "state":
        state_estimation = True
        parameter_estimation = False
    elif args.estim == "parameter":
        state_estimation = False
        parameter_estimation = True

    algo = args.algo

    # create out_folder for saving plots:
    out_folder = args.out_path
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)


    # ----------------------------------------- create synthetic SV dataset --------------------------------------
    # values paper Jimmy: 0.8,O.1,1.
    # other params: 0.91, 1.0, 0.5
    # do with 1000 observations.
    seq_len = args.seq_len

    if args.data_path is None:
        scale0 = alpha / np.sqrt(1 - alpha ** 2)
        X0 = np.random.normal(scale=scale0)
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

        print("saving data in .npy file...")
        np.save(os.path.join(out_folder, 'observations.npy'), observations)
        np.save(os.path.join(out_folder, 'states.npy'), states)

    else:
        print("uploading observations and states")
        observations = np.load(os.path.join(args.data_path, "observations.npy"))
        states = np.load(os.path.join(args.data_path, "states.npy"))



    print("OBSERVATIONS", observations)


    observations = observations[np.newaxis, :]
    observations = np.repeat(observations, num_particles, axis=0)
    # observations = torch tensor of size T.
    observations = torch.tensor(observations, dtype=torch.float32)

    # ---------------------------------------------------------------------------------------------------------------------------- #

    # Create bootstrap filter with init params and number of particles
    bt_filter = SVBootstrapFilter(num_particles, init_params)

    # ---------------------------- Test PMS ------------------------------------------------------------------------------------------

    # ---------------------Test state estimation ----------------------------------------------------------------------------------#
    if state_estimation:
        index_states = [1, 24, 49, 99]
        # index_states = [23]
        num_trials = args.n_trials
        states_estims = np.zeros((len(index_states), num_trials))
        results = dict.fromkeys(index_states)
        for iter, index_state in enumerate(index_states):

            print("INDEX STATE:", index_state)

            # state_estims = np.zeros(num_trials)
            # create SVBackward IS smoothing with number of backward samples, out_folder, logger.
            if algo == 'BIS':
                smoother = SVBackwardISSmoothing(backward_samples=backward_samples, observations=observations,
                                             bootstrap_filter=bt_filter, index_state=index_state)
            elif algo == 'PMS':
                smoother = PoorManSmoothing(observations=observations,
                                                 bootstrap_filter=bt_filter, index_state=index_state)

            bt_filter.update_SV_params([alpha, sigma, beta])

            print("params of SV model:", smoother.bootstrap_filter.params)


            for trials in range(num_trials):
                start_time = time.time()
                if algo == "BIS":
                    state_estim = -smoother.estimate_conditional_expectation_of_function(params=[alpha, sigma, beta])
                elif algo == "PMS":
                    smoother.save_smoothing_elements()
                    state_estim = -smoother.compute_expectation_from_saved_elements([alpha, sigma, beta])
                print("time for one estimation", time.time()-start_time)
                states_estims[iter, trials] = state_estim

            print("TRUE STATE", states[index_state])
            print('MEAN STATE PREDICTED', np.mean(states_estims[iter]))

            mean_mse = np.mean(0.5 * np.square(states[index_state] - states_estims[iter]))
            var_mse = np.var(0.5 * np.square(states[index_state] - states_estims[iter]))
            results[index_state] = str(mean_mse) + "+/-" + str(var_mse)
            print("MEAN MSE", mean_mse)
            print("VAR MSE", var_mse)

            print("-" * 80)

        out_folder = os.path.join(out_folder, "{}_state_estimation_P{}_{}J_{}S".format(algo, num_particles, backward_samples,
                                                                                    seq_len))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        true_states = [states[index_state] for index_state in index_states]
        plot_state_estimation(states_estims, true_states, index_states,
                                  os.path.join(out_folder, "plot_states_estimation.png"))
        with open(os.path.join(out_folder, "results.json"), "w") as f:
            json.dump(results, f)

    ######################### ----------------------------------------------------- #################################"

    ######################### ---- Parameter Estimation with EM algo --------------- ###################################
    elif parameter_estimation:
        optim_method = 'Powell'  # optimizers tried: 'BFGS', 'Nelder-Mead', 'Powell', 'L-BFGS-B'
        # BFGS, L-BFGS -> params do not move.
        out_folder = os.path.join(out_folder,
                                  "{}_EM_{}_{}P_{}J_{}S_params-{}-{}-{}-initparams-{}-{}-{}".format(algo, optim_method, num_particles, backward_samples,
                                                                              seq_len, alpha, sigma, beta, init_params[0],
                                                                              init_params[1], init_params[2]))
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        print("saving data in .npy file...")
        np.save(os.path.join(out_folder, 'observations.npy'), observations)
        np.save(os.path.join(out_folder, 'states.npy'), states)

        maxiter = 300
        options = {'maxiter': maxiter, 'maxfev': None, 'disp': True, 'return_all': True, 'initial_simplex': None,
                   'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False}

        print("INIT PARAMS: {}".format(init_params))

        if algo == "BIS":
            smoother = SVBackwardISSmoothing(backward_samples=backward_samples, observations=observations,
                                         bootstrap_filter=bt_filter)
        elif algo == "PMS":
            smoother = PoorManSmoothing(bootstrap_filter=bt_filter, observations=observations)

        bt_filter.update_SV_params(init_params)

        # EM algo.
        expectations_results, list_params = [], [init_params]
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

            print("time for one EM", time.time()-start_time)

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
        plot_evol_params(list_params=list_params, true_params=[alpha, rho, mu],
                         out_file=os.path.join(out_folder, "plot_evol_params.png"))

        params_star = dict(zip(["alpha", "sigma", "beta"], [str(i) for i in results.x]))
        with open(os.path.join(out_folder, "params_star.json"), 'w') as f:
            json.dump(params_star, f)
        np.save(os.path.join(out_folder, "list_params.npy"), np.array(list_params))

        print("done")

# Loop over number of iterations for the EM algo.
# estimate the E-step with backwardIS smoothing (get an estimation of the log likelihood)

# Max-step: argmax over  parameters
# update bootstrap filter params.

# class SVBackwardISSmoothing(SmoothingAlgo):
#     def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function, out_folder,
#                  index_state=0, save_elements=False, logger=None)
