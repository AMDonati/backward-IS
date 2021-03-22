import torch
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.utils import resample, resample_all_seq
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from train.utils import create_logger


class SmoothingAlgo:
    def __init__(self, bootstrap_filter, observations, states, estimation_function, out_folder,
                 index_state=0):
        # '''
        # :param bootstrap_filter: Class Implementing a Bootstrap filter algorithm.
        # :param observations: sequence of observations generated by a stochastic RNN: tensor of shape (num_samples=B, num_particles, seq_len, output_size)
        # :param states: sequence of hidden states generated by a stochastic RNN: tensor of shape (num_samples, num_particles, seq_len, hidden_size)
        # :param backward_samples: number of backward_samples for the Backward IS Smoothing algo.
        # :param estimation_function: Fonction to estimate: in our case $mathbb[E][X_0|Y_{0:n}]$
        # '''
        self.bootstrap_filter = bootstrap_filter
        self.rnn = bootstrap_filter.rnn
        self.observations = observations  # Tensor of shape (B, particles, seq_len, output_size)
        self.states = states  # Tensor of shape (B, particles, seq_len, hidden_size)
        self.num_particles = self.bootstrap_filter.num_particles
        self.estimation_function = estimation_function
        self.index_state = index_state
        self.seq_len = self.observations.size(-2)

        self.logger = self.create_logger(out_folder)

    def create_logger(self, out_folder):
        out_file_log = os.path.join(out_folder, 'debug_log.log')
        logger = create_logger(out_file_log=out_file_log)
        return logger

    def init_particles(self):
        sigma_init = self.rnn.sigma_init
        self.ancestors = torch.normal(
            mean=torch.zeros(self.states.size(0), self.num_particles, self.states.size(-1)),
            std=sigma_init ** (1 / 2) * torch.ones(self.states.size(0), self.num_particles,
                                                   self.states.size(-1)))
        # self.ancestors = self.states[:, :, 0, :].repeat(1, self.num_particles, 1)  # (B, num_particles, hidden_size)
        self.trajectories = self.ancestors.unsqueeze(-2)
        self.filtering_weights = self.bootstrap_filter.compute_filtering_weights(hidden=self.ancestors,
                                                                                 observations=self.observations[:,
                                                                                              :, 0,
                                                                                              :])  # decide if take $Y_0 of $Y_1$
        self.past_tau = torch.zeros(self.states.size(0), self.num_particles, self.states.size(-1))
        self.new_tau = self.past_tau
        self.taus = []
        self.all_IS_weights = []

    def plot_estimation_versus_state(self, phi, out_folder):
        estimation = phi[0].squeeze().cpu().numpy()
        true_state = self.states[0, :, self.index_state, :].squeeze().cpu().numpy()
        error = true_state - estimation
        x = np.linspace(1, phi.size(-1), phi.size(-1))
        plt.scatter(x, estimation, marker='o', color='cyan', lw=2,
                    label='estimation of X_{}'.format(self.index_state))
        plt.scatter(x, true_state, marker='x', color='green', lw=2,
                    label='ground-truth for X_{}'.format(self.index_state))
        plt.plot(x, error, 'red', lw=1, linestyle='dashed', label='error for X_{}'.format(self.index_state))
        plt.legend(fontsize=10)
        out_file = "plot_estimation_vs_true_X{}".format(self.index_state)
        plt.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def plot_multiple_runs(self, phis_backward, phis_pms, out_folder, num_runs):
        fig, ax = plt.subplots(figsize=(12, 10))
        phis_backward = torch.stack(phis_backward, dim=0)
        phi_backward_mean = phis_backward.mean(0).squeeze().cpu().numpy()
        phis_pms = torch.stack(phis_pms, dim=0)
        phi_pms_mean = phis_pms.mean(0).squeeze().cpu().numpy()
        phis_backward = phis_backward[:, 0, :].cpu().numpy()
        phis_pms = phis_pms[:, 0, :].cpu().numpy()
        true_state = self.states[0, :, self.index_state, :].squeeze().cpu().numpy()
        num_dim = phis_backward.shape[-1]
        x = np.linspace(1, num_dim, num_dim)
        xx = np.linspace(1 + 0.2, num_dim + 0.2, num_dim)
        for i in range(phis_backward.shape[0]):
            label_backward = 'BACKWARD IS: estimation of X_{}'.format(self.index_state) if i == 0 else None
            label_pms = 'PMS: estimation of X_{}'.format(self.index_state) if i == 0 else None
            ax.scatter(x, phis_backward[i], marker='o', color='cyan', lw=1, label=label_backward)
            ax.scatter(xx, phis_pms[i], marker='o', color='pink', lw=1, label=label_pms)
        ax.scatter(x, phi_backward_mean, marker='o', color='blue',
                   label='BACKWARD IS: mean estimation of X_{}'.format(self.index_state))
        ax.scatter(xx, phi_pms_mean, marker='o', color='red',
                   label='PMS: mean estimation of X_{}'.format(self.index_state))
        ax.scatter(x, true_state, marker='x', color='green', lw=1,
                   label='ground-truth for X_{}'.format(self.index_state))
        # plt.plot(x, error, 'red', lw=1, linestyle='dashed', label='error for X_{}'.format(self.index_state))
        ax.legend(fontsize=10)
        out_file = "plot_estimation_vs_true_X{}_{}runs".format(self.index_state, num_runs)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def boxplots_error(self, errors_backward, errors_pms, out_folder, num_runs):
        errors_backward = torch.stack(errors_backward, dim=0).squeeze(1).cpu().numpy()
        errors_pms = torch.stack(errors_pms, dim=0).squeeze(1).cpu().numpy()
        errors_dim_backward = [errors_backward[:, dim] for dim in range(errors_backward.shape[-1])]
        errors_dim_pms = [errors_pms[:, dim] for dim in range(errors_pms.shape[-1])]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.boxplot(errors_dim_backward)
        ax2.boxplot(errors_dim_pms)
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        out_file = "error_boxplot_X{}_{}runs".format(self.index_state, num_runs)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def plot_particles_all_k(self, particles_backward, weights_backward, out_folder, num_runs, particles_pms=None,
                             weights_pms=None):
        particles_backward = torch.stack(particles_backward, dim=0).cpu().squeeze(1).numpy()
        particles_pms = torch.stack(particles_pms, dim=0).cpu().squeeze(1).numpy()
        weights_backward = torch.stack(weights_backward, dim=0).cpu().squeeze(1).numpy()
        weights_pms = torch.stack(weights_pms, dim=0).cpu().squeeze(1).numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        seq_len = particles_backward.shape[0]
        parts_1 = [particles_backward[i, :, 0] for i in range(seq_len)]
        parts_2 = [particles_backward[i, :, 1] for i in range(seq_len)]
        parts_1_pms = [particles_pms[i, :, 0] for i in range(seq_len)]
        parts_2_pms = [particles_pms[i, :, 1] for i in range(seq_len)]
        bplot1 = ax1.boxplot(parts_1, patch_artist=True)
        # bplot1_pms = ax1.boxplot(parts_1_pms, patch_artist=True)
        for patch in bplot1['boxes']:
            patch.set_facecolor('blue')
        # for patch in bplot1_pms['boxes']:
        #     patch.set_facecolor('red')
        bplot2 = ax2.boxplot(parts_2, patch_artist=True)
        # bplot2_pms = ax2.boxplot(parts_2_pms, patch_artist=True)
        for patch in bplot2['boxes']:
            patch.set_facecolor('blue')
        # for patch in bplot2_pms['boxes']:
        #     patch.set_facecolor('red')
        x = np.linspace(1, seq_len, seq_len)
        xx = np.linspace(1 - 0.02, seq_len - 0.02, seq_len)
        xxx = np.linspace(1 + 0.02, seq_len + 0.02, seq_len)
        if seq_len < self.states.size(-2):
            states = self.states[:, :, :seq_len, :]
        else:
            states = self.states
        for m in range(particles_backward.shape[-2]):
            ax1.scatter(xx, particles_backward[:, m, 0], s=weights_backward[:, m] * 100, color='blue')
            ax2.scatter(xx, particles_backward[:, m, 1], s=weights_backward[:, m] * 100, color='blue')
            ax1.scatter(xxx, particles_pms[:, m, 0], s=weights_pms[:, m] * 100, color='red')
            ax2.scatter(xxx, particles_pms[:, m, 1], s=weights_pms[:, m] * 100, color='red')
        ax1.scatter(x, states[:, :, :, 0].squeeze().cpu().numpy(), color='green', marker='x')
        ax2.scatter(x, states[:, :, :, 1].squeeze().cpu().numpy(), color='green', marker='x')
        out_file = "particles_allseq_{}runs".format(num_runs)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def plot_trajectories_pms(self, trajectories, out_folder):
        trajectories = torch.stack(trajectories, dim=0).cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        for s in range(trajectories.shape[0]):
            pass

    def compute_mse_phi_X0(self, phi):
        # ''':param phi: estimation of $mathbb[E][X_0|Y_{0:n}]$: tensor of shape (B, hidden size).
        # '''
        criterion = nn.MSELoss(reduction='none')
        error = phi - self.states[:, :, self.index_state, :].mean(dim=1)
        loss = criterion(phi, self.states[:, :, self.index_state, :].mean(dim=1))
        return loss.mean(), error


class RNNBackwardISSmoothing(SmoothingAlgo):
    def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function, out_folder,
                 index_state=0, save_elements=False):
        # '''
        # :param bootstrap_filter: Class Implementing a Bootstrap filter algorithm.
        # :param observations: sequence of observations generated by a stochastic RNN: tensor of shape (num_samples=B, num_particles, seq_len, output_size)
        # :param states: sequence of hidden states generated by a stochastic RNN: tensor of shape (num_samples, num_particles, seq_len, hidden_size)
        # :param backward_samples: number of backward_samples for the Backward IS Smoothing algo.
        # :param estimation_function: Fonction to estimate: in our case $mathbb[E][X_0|Y_{0:n}]$
        # '''
        super(RNNBackwardISSmoothing, self).__init__(bootstrap_filter=bootstrap_filter, observations=observations,
                                                     states=states, estimation_function=estimation_function,
                                                     out_folder=out_folder,
                                                     index_state=index_state)
        self.backward_samples = backward_samples
        self.save_elements = save_elements
        self.init_particles()

    def update_tau(self, ancestors, particle, backward_indices, IS_weights, k):
        # '''
        # :param ancestors: ancestors particles $\xi_{k-1}^Jk$ sampled with backward_indices. tensor of shape (B, backward_samples, hidden_size)
        # :param particle: $\xi_k^l$ (here not used in the formula of the estimation function): tensor of shape (B, 1, hidden_size)
        # :param backward_indices: $J_k(j)$: tensor of shape (B, backward_samples)
        # :param IS_weights: normalized importance sampling weights: tensor of shape (B, backward_samples)
        # :param k: current timestep.
        # '''
        # '''update $\tau_k^l from $\tau_{k-1}^l, $w_{k-1]^l, $\xi_{k-1}^Jk$ and from Jk(j), \Tilde(w)(l,j) for all j in 0...backward samples'''

        resampled_tau = resample(self.past_tau, backward_indices)  # (B,backward_samples, hidden_size)
        new_tau_element = IS_weights * (resampled_tau + self.estimation_function(k=k, X=ancestors,
                                                                                 index=self.index_state))  # (B, backward_samples, hidden_size)
        new_tau = new_tau_element.sum(1)
        return new_tau

    def estimate_conditional_expectation_of_function(self):
        self.init_particles()
        with torch.no_grad():
            # for loop on time
            for k in range(self.seq_len - 1):
                self.logger.info(
                    "---------------------------------------------- TIMESTEP {}-------------------------------------------------------".format(
                        k))
                # Run bootstrap filter at time k
                self.old_filtering_weights = self.filtering_weights
                self.past_tau = self.new_tau
                (self.particles, _), self.filtering_weights = self.bootstrap_filter.get_new_particle(
                    observation=self.observations[:, :, k, :], next_observation=self.observations[:, :, k + 1, :],
                    hidden=self.ancestors, weights=self.old_filtering_weights)
                var_1 = torch.var(self.particles[:,:,0], dim=1).squeeze().numpy()
                var_2 = torch.var(self.particles[:,:,1], dim=1).squeeze().numpy()
                self.logger.info("BACKWARD IS - PARTICLES VARIABILITY - dim 1: {}".format(var_1.item()))
                self.logger.info("BACKWARD IS - PARTICLES VARIABILITY - dim 2: {}".format(var_2.item()))
                # Backward Simulation
                # For loop of number of particles
                new_taus, all_is_weights = [], []
                for l in range(self.num_particles):
                    # Select one particle.
                    particle = self.particles[:, l, :].unsqueeze(dim=1)  # shape (B, 1, hidden)
                    # A. Get backward Indice J from past filtering weights
                    backward_indices = torch.multinomial(self.old_filtering_weights,
                                                         self.backward_samples)  # shape (B, J)
                    # B. Select Ancestor with J.
                    ancestors = resample(self.ancestors, backward_indices)  # shape (B, J, hidden) # ok function resample checked.
                    # C. Compute IS weights with Ancestor & Particle.
                    is_weights = self.rnn.estimate_transition_density(ancestor=ancestors, particle=particle,
                                                                      previous_observation=self.observations[:, :, k,
                                                                                           :])
                    # End for
                    # compute $\tau_k^l$ with all backward IS weights, ancestors, current particle & all backward_indices.
                    new_tau = self.update_tau(ancestors=ancestors, particle=particle, backward_indices=backward_indices,
                                              IS_weights=is_weights.unsqueeze(-1), k=k)
                    new_taus.append(new_tau)
                    all_is_weights.append(is_weights)
                # End for
                self.new_tau = torch.stack(new_taus, dim=1)  # shape (B, num_particles, hidden_size)
                var_1 = torch.var(self.new_tau[:, :, 0], dim=1).squeeze().numpy()
                var_2 = torch.var(self.new_tau[:, :, 1], dim=1).squeeze().numpy()
                self.logger.info("TAU VARIABILITY - dim 1: {}".format(var_1.item()))
                self.logger.info("TAU VARIABILITY - dim 2: {}".format(var_2.item()))
                self.ancestors = self.particles
                # if self.save_elements:
                #     self.taus.append(self.new_tau)
                #     self.logger.info("-------------TAU------------------------")
                #     self.logger.info(self.new_tau.squeeze().cpu().numpy())
                #     self.logger.info("-------------IS WEIGHTS------------------------")
                #     self.all_IS_weights.append(torch.stack(all_is_weights, dim=1))
                #     self.logger.info(torch.stack(all_is_weights, dim=1).squeeze().cpu().numpy())
                #     self.logger.info(
                #         "-------------------------------------------------------------------------------------------------------------------------------")
            # End for
            # Compute $\phi_n$ with last filtering weights and last $tau$.
            phi_element = self.filtering_weights.unsqueeze(-1) * self.new_tau  # TODO: should be old_filtering_weights ?
            phi = phi_element.sum(1)  # shape (B, hidden_size)
        return phi, (self.new_tau, self.filtering_weights)

    def debug_elements(self, data_path):
        if len(self.taus) > 0:
            taus = torch.stack(self.taus, dim=2).cpu().numpy()
            np.save(os.path.join(data_path, "taus_X{}.npy".format(self.index_state)), taus)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            x = np.linspace(0, taus.shape[-2] - 1, taus.shape[-2])
            y1 = taus[0, :, :, 0].mean(0)
            y2 = taus[0, :, :, 1].mean(0)
            ax1.plot(x, taus[0, :, :, 0].mean(0), label='dim 0 of mean tau')
            yy1 = [taus[:, :, i, 0].squeeze() for i in range(taus.shape[-2])]
            yy2 = [taus[:, :, i, 1].squeeze() for i in range(taus.shape[-2])]
            ax1.boxplot(yy1)
            for xs, ys in zip(x, y1):
                label = "{:.2f}".format(ys)
                ax1.annotate(label,  # this is the text
                             (xs, ys),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 5),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center
            ax2.plot(x, taus[0, :, :, 1].mean(0), label='dim 1 of mean tau')
            ax2.boxplot(yy2)
            for xs, ys in zip(x, y2):
                label = "{:.2f}".format(ys)
                ax2.annotate(label,  # this is the text
                             (xs, ys),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center
            out_file = "debug_taus_X{}".format(self.index_state)
            fig.savefig(os.path.join(data_path, out_file))
            plt.close()
        if len(self.all_IS_weights) > 0:
            all_is_weights = torch.stack(self.all_IS_weights, dim=2).cpu().numpy()
            np.save(os.path.join(data_path, "IS_weights_X{}.npy".format(self.index_state)), all_is_weights)


class PoorManSmoothing(SmoothingAlgo):
    def __init__(self, bootstrap_filter, observations, states, estimation_function, out_folder,
                 index_state=0):
        # '''
        # :param bootstrap_filter: Class Implementing a Bootstrap filter algorithm.
        # :param observations: sequence of observations generated by a stochastic RNN: tensor of shape (num_samples=B, num_particles, seq_len, output_size)
        # :param states: sequence of hidden states generated by a stochastic RNN: tensor of shape (num_samples, num_particles, seq_len, hidden_size)
        # :param backward_samples: number of backward_samples for the Backward IS Smoothing algo.
        # :param estimation_function: Fonction to estimate: in our case $mathbb[E][X_0|Y_{0:n}]$
        # '''
        super(PoorManSmoothing, self).__init__(bootstrap_filter=bootstrap_filter, observations=observations,
                                               states=states, estimation_function=estimation_function,
                                               out_folder=out_folder,
                                               index_state=index_state)
        self.init_particles()

    def estimate_conditional_expectation_of_function(self):
        self.init_particles()
        with torch.no_grad():
            # for loop on time
            for k in range(self.seq_len - 1):
                # Selection: resample all past trajectories with current indice i_t
                self.old_filtering_weights = self.filtering_weights
                i_t = torch.multinomial(self.old_filtering_weights, num_samples=self.num_particles)
                resampled_trajectories = resample_all_seq(self.trajectories, i_t=i_t)
                ancestor = resampled_trajectories[:, :, k, :]  # get resampled ancestor $\xi_{k-1}$
                # Mutation: Run bootstrap filter at time k to get new particle without resampling
                (self.particles, _), self.filtering_weights = self.bootstrap_filter.get_new_particle(
                    observation=self.observations[:, :, k, :], next_observation=self.observations[:, :, k + 1, :],
                    hidden=ancestor, weights=self.old_filtering_weights, resampling=False)
                var_1 = torch.var(self.particles[:, :, 0], dim=1).squeeze().numpy()
                var_2 = torch.var(self.particles[:, :, 1], dim=1).squeeze().numpy()
                self.logger.info("PMS - PARTICLES VARIABILITY - dim 1: {}".format(var_1.item()))
                self.logger.info("PMS - PARTICLES VARIABILITY - dim 2: {}".format(var_2.item()))
                # append resampled trajectories to new particle
                self.trajectories = torch.cat([resampled_trajectories, self.particles.unsqueeze(-2)], dim=-2)
            h_k_elements = torch.stack(
                [self.estimation_function(k=k, X=self.trajectories[:, :, k, :], index=self.index_state) for k in
                 range(self.seq_len)], dim=-2)  # (B,P,S,hidden_size)
            h_n = h_k_elements.sum(-2)
            phi_element = self.filtering_weights.unsqueeze(-1) * h_n
            phi = phi_element.sum(1)  # (B, hidden)
            return phi, (h_n, self.filtering_weights, self.trajectories)
