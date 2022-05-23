import numpy as np
import torch
from smc.BootstrapFilter import SVBootstrapFilter
from smc.BackwardIS_smoother import SVBackwardISSmoothing

if __name__ == '__main__':
    # hparams
    num_particles = 10
    backward_samples = 4
    init_params = [0.1, 0.2, 0.3]

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

    phis, (last_tau, last_filtering_weights) = smoother.estimate_conditional_expectation_of_function()

    expectation = (last_tau.squeeze()*last_filtering_weights).sum().numpy()


    print("done")

# Loop over number of iterations for the EM algo.
    # estimate the E-step with backwardIS smoothing (get an estimation of the log likelihood)

    # Max-step: argmax over  parameters
    # update bootstrap filter params. 

# class SVBackwardISSmoothing(SmoothingAlgo):
#     def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function, out_folder,
#                  index_state=0, save_elements=False, logger=None)