import numpy as np
import torch
from smc.BootstrapFilter import SVBootstrapFilter
from smc.BackwardIS_smoother import SVBackwardISSmoothing
import scipy.optimize as opt

if __name__ == '__main__':
    # hparams
    num_particles = 10
    backward_samples = 4
    init_params = [0.1, 0.2, 0.3]
    n_iter = 10

    # upload dataset
    observations = np.array([0.01*i for i in range(12)])
    observations = observations[np.newaxis, :]

    observations = np.repeat(observations, num_particles, axis=0)

    # observations = torch tensor of size T.
    observations = torch.tensor(observations, dtype=torch.float32)

    # Create bootstrap filter with init params and number of particles
    bt_filter = SVBootstrapFilter(num_particles, init_params)

    # create SVBackward IS smoothing with number of backward samples, out_folder, logger.
    smoother = SVBackwardISSmoothing(backward_samples=backward_samples, observations=observations, bootstrap_filter=bt_filter)

    print("INIT PARAMS: {}".format(init_params))
    for iter in range(n_iter):

        expectation = smoother.estimate_conditional_expectation_of_function(bt_filter.params)
        print("negative log-likelihood at iter {}: {}".format(iter, expectation))

        results = opt.minimize(fun=smoother.estimate_conditional_expectation_of_function, x0=bt_filter.params)

        print("new params: {}".format(results.x))
        bt_filter.update_SV_params(results.x)

        print("-"*30)

    print("done")

# Loop over number of iterations for the EM algo.
    # estimate the E-step with backwardIS smoothing (get an estimation of the log likelihood)

    # Max-step: argmax over  parameters
    # update bootstrap filter params. 

# class SVBackwardISSmoothing(SmoothingAlgo):
#     def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function, out_folder,
#                  index_state=0, save_elements=False, logger=None)