import torch
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.utils import resample, resample_all_seq
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from train.utils import create_logger
import random
import time


class SmoothingAlgo:
    def __init__(self, bootstrap_filter, observations, states, estimation_function, out_folder,
                 index_state=0, logger=None):
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
        if logger is None:
            self.logger = self.create_logger(out_folder)
        else:
            self.logger = logger


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

    def boxplots_error(self, errors_backward, errors_pms, out_folder, num_runs, index_states):
        backward_1 = [err[:,0] for err in errors_backward]
        backward_2 = [err[:,1] for err in errors_backward]
        pms_1 = [err[:, 0] for err in errors_pms]
        pms_2 = [err[:, 1] for err in errors_pms]
        mean_backward_1 = [np.mean(err) for err in backward_1]
        mean_backward_2 = [np.mean(err) for err in backward_2]
        mean_pms_1 = [np.mean(err) for err in pms_1]
        mean_pms_2 = [np.mean(err) for err in pms_2]
        xx = np.linspace(1, len(pms_1), len(pms_1))
        positions_pms = np.linspace(1+0.2, len(pms_1)+0.2, len(pms_1))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 12))
        labels = ["X_{}".format(k) for k in index_states]
        bb1_b = ax1.boxplot(backward_1, patch_artist=True, widths=0.25, sym="")
        bb1_p = ax1.boxplot(pms_1, patch_artist=True, positions=positions_pms, manage_ticks=False, widths=0.3, sym="")
        bb2_b = ax2.boxplot(backward_2, patch_artist=True, widths=0.25, sym="")
        bb2_p = ax2.boxplot(pms_2, patch_artist=True, positions=positions_pms, manage_ticks=False, widths=0.3, sym="")
        for patch in bb1_b['boxes']:
            patch.set_facecolor('blue')
        for patch in bb2_b['boxes']:
            patch.set_facecolor('blue')
        for patch in bb1_p['boxes']:
            patch.set_facecolor('red')
        for patch in bb2_p['boxes']:
            patch.set_facecolor('red')
        plt.sca(ax1)
        plt.xticks(labels=labels, ticks=xx)
        ax1.plot(xx, mean_backward_1, color='blue', label='backward IS Smoother')
        ax1.plot(positions_pms, mean_pms_1, color='red', label='PMS Smoother')
        ax2.plot(xx, mean_backward_2, color='blue')
        ax2.plot(positions_pms, mean_pms_2, color='red')
        plt.sca(ax2)
        plt.xticks(labels=labels, ticks=xx)
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        ax1.grid('on')
        ax2.grid('on')
        ax1.set_title('boxplot of estimation error')
        out_file = "error_boxplot_{}runs_{}particles_{}J".format(num_runs, self.num_particles, self.backward_samples)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def boxplots_loss(self, loss_backward, loss_pms, out_folder, num_runs, index_states):
        xx = np.linspace(1, len(loss_pms), len(loss_pms))
        positions_pms = np.linspace(1+0.2, len(loss_pms)+0.2, len(loss_pms))
        fig, ax = plt.subplots(figsize=(25, 10))
        labels = ["X_{}".format(k) for k in index_states]
        bb1_b = ax.boxplot(loss_backward, patch_artist=True, widths=0.25, sym="")
        bb1_p = ax.boxplot(loss_pms, patch_artist=True, positions=positions_pms, manage_ticks=False, widths=0.25, sym="")
        mean_backward = [np.mean(l) for l in loss_backward]
        mean_pms = [np.mean(l) for l in loss_pms]
        for patch in bb1_b['boxes']:
            patch.set_facecolor('blue')
        for patch in bb1_p['boxes']:
            patch.set_facecolor('red')
        ax.plot(xx, mean_backward, color='blue', label='Backward IS Smoother')
        ax.plot(positions_pms, mean_pms, color='red', label='PMS Smoother')
        plt.xticks(ticks=xx, labels=labels)
        ax.legend(fontsize=12)
        ax.grid('on')
        ax.set_title('Boxplot of squared error', fontsize=18)
        out_file = "loss_boxplot_{}runs_{}particles_{}J".format(num_runs, self.num_particles, self.backward_samples)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def plot_particles_all_k(self, particles_backward, weights_backward, out_folder, num_runs, index_states, particles_pms=None,
                             weights_pms=None):
        particles_backward = torch.stack(particles_backward, dim=0).cpu().squeeze(1).numpy()
        particles_pms = particles_pms.cpu().squeeze(0).numpy().transpose(1,0,2)
        weights_backward = torch.stack(weights_backward, dim=0).cpu().squeeze(1).numpy()
        weights_pms = weights_pms.repeat(weights_backward.shape[0], 1).numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        labels = ["X_{}".format(k) for k in index_states]
        seq_len = particles_backward.shape[0]
        x = np.linspace(1, seq_len, seq_len)
        xx = np.linspace(1 - 0.02, seq_len - 0.02, seq_len)
        xxx = np.linspace(1 + 0.02, seq_len + 0.02, seq_len)
        if seq_len < self.states.size(-2):
            states = self.states[:, :, index_states, :]
        else:
            states = self.states
        for m in range(particles_backward.shape[-2]):
            ax1.scatter(xx, particles_backward[:, m, 0], s=weights_backward[:, m] * 100, color='blue')
            ax2.scatter(xx, particles_backward[:, m, 1], s=weights_backward[:, m] * 100, color='blue')
            ax1.scatter(xxx, particles_pms[:, m, 0], s=weights_pms[:, m] * 100, color='red')
            ax2.scatter(xxx, particles_pms[:, m, 1], s=weights_pms[:, m] * 100, color='red')
        ax1.scatter(x, states[:, :, :, 0].squeeze().cpu().numpy(), color='green', marker='x')
        ax2.scatter(x, states[:, :, :, 1].squeeze().cpu().numpy(), color='green', marker='x')
        plt.sca(ax1)
        plt.xticks(ticks=xx, labels=labels)
        plt.sca(ax2)
        plt.xticks(ticks=xx, labels=labels)
        out_file = "particles_allseq_{}particles_{}runs_{}J".format(self.num_particles, num_runs, self.backward_samples)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def plot_trajectories_pms(self, trajectories, out_folder):
        trajectories = trajectories.squeeze().cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        x = np.linspace(1, trajectories.shape[1], trajectories.shape[1])
        num_part = [np.unique(trajectories[:, k, :], axis=0).shape[0] for k in range(trajectories.shape[1])]
        for p in range(trajectories.shape[0]):
            label1 = "trajectory for  dim 0" if p == 0 else None
            ax1.scatter(x, trajectories[p,:,0], label=label1, s=7)
        ax2.plot(x, num_part, label='number of unique particles')
        ax1.legend(loc='upper center')
        ax2.legend(loc='upper center')
        out_file = "pms_trajectories_{}particles".format(self.num_particles)
        fig.savefig(os.path.join(out_folder, out_file))
        plt.close()

    def compute_mse_phi_X0(self, phi):
        # ''':param phi: estimation of $mathbb[E][X_0|Y_{0:n}]$: tensor of shape (B, hidden size).
        # '''
        criterion = nn.MSELoss(reduction='none')
        error = phi - self.states[:, :, self.index_state, :].mean(dim=1) # hidden_size
        loss = criterion(phi, self.states[:, :, self.index_state, :].mean(dim=1))
        return loss.mean().item(), error


class RNNBackwardISSmoothing(SmoothingAlgo):
    def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function, out_folder,
                 index_state=0, save_elements=False, logger=None):
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
                                                     index_state=index_state, logger=logger)
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

        resampled_tau = resample(self.past_tau.repeat(backward_indices.size(0), 1, 1), backward_indices)  # (B,backward_samples, hidden_size)
        new_tau_element = IS_weights * (resampled_tau + self.estimation_function(k=k, X=ancestors,
                                                                                 index=self.index_state))  # (B, backward_samples, hidden_size)
        new_tau = new_tau_element.sum(1)
        #print("NEW TAU", new_tau[:, 0])
        return new_tau

    def estimate_conditional_expectation_of_function(self):
        start_time = time.time()
        self.init_particles()
        mses, errors, phis = [], [], []
        with torch.no_grad():
            # for loop on time
            for k in range(self.seq_len - 1):
                # Run bootstrap filter at time k
                self.old_filtering_weights = self.filtering_weights
                self.past_tau = self.new_tau
                (self.particles, _), self.filtering_weights = self.bootstrap_filter.get_new_particle(
                    observation=self.observations[:, :, k, :], next_observation=self.observations[:, :, k + 1, :],
                    hidden=self.ancestors, weights=self.old_filtering_weights)

                # Backward Simulation
                # A. Get backward Indice J from past filtering weights
                backward_indices = torch.multinomial(self.old_filtering_weights,
                                                     self.num_particles * self.backward_samples, replacement=True)
                backward_indices = backward_indices.view(self.num_particles, self.backward_samples)  # shape (B, J)

                # B. Select Ancestor with J.
                ancestors = resample(self.ancestors.repeat(backward_indices.size(0), 1, 1),
                                     backward_indices)  # shape (P, J, hidden) # ok function resample checked.
                # C. Compute IS weights with Ancestor & Particle.
                is_weights = self.rnn.estimate_transition_density(ancestor=ancestors, particle=self.particles.squeeze(),
                                                                  previous_observation=self.observations[:, :, k,
                                                                                       :]) # shape (B,J)
                # End for
                # compute $\tau_k^l$ with all backward IS weights, ancestors, current particle & all backward_indices.
                new_tau = self.update_tau(ancestors=ancestors, particle=self.particles, backward_indices=backward_indices,
                                          IS_weights=is_weights.unsqueeze(-1), k=k)
                self.new_tau = new_tau
                self.ancestors = self.particles
                self.taus.append(self.new_tau)
                # Compute online estimation: $\mathbb[E][X_k|Y_{0:j}]$
                phi_element = self.filtering_weights.unsqueeze(-1) * self.new_tau
                phi = phi_element.sum(1)  # shape (B, hidden_size)
                mse, error = self.compute_mse_phi_X0(phi)
                mses.append(mse)
                errors.append(error)
                phis.append(phi)
        total_time = time.time() - start_time
        self.logger.info("TIME FOR ONE BACKWARD IS - num particles {} - backward samples {}- seq len {}: {}".format(self.num_particles, self.backward_samples, self.seq_len, total_time))
        return (mses, errors), phis, (self.new_tau, self.filtering_weights)

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
                 index_state=0, logger=None):
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
                                               index_state=index_state, logger=logger)
        self.init_particles()

    def estimate_conditional_expectation_of_function(self):
        start_time = time.time()
        self.init_particles()
        with torch.no_grad():
            # for loop on time
            indices_matrix, particles_seq = [], []
            mses, errors, phis = [], [], []
            particles_seq.append(self.ancestors)
            for k in range(self.seq_len - 1):
                # Selection: resample all past trajectories with current indice i_t
                self.old_filtering_weights = self.filtering_weights
                i_t = torch.multinomial(self.old_filtering_weights, num_samples=self.num_particles, replacement=True)
                indices_matrix.append(i_t.cpu().squeeze())
                resampled_trajectories = resample_all_seq(self.trajectories, i_t=i_t)
                ancestor = resampled_trajectories[:, :, k, :]  # get resampled ancestor $\xi_{k-1}$
                # Mutation: Run bootstrap filter at time k to get new particle without resampling
                (self.particles, _), self.filtering_weights = self.bootstrap_filter.get_new_particle(
                    observation=self.observations[:, :, k, :], next_observation=self.observations[:, :, k + 1, :],
                 hidden=ancestor, weights=self.old_filtering_weights, resampling=False)
                particles_seq.append(self.particles)
                # append resampled trajectories to new particle
                self.trajectories = torch.cat([resampled_trajectories, self.particles.unsqueeze(-2)], dim=-2)
                # get mse, error for online estimation:
                error, mse, phi = self.get_error(k)
                errors.append(error)
                mses.append(mse)
                phis.append(phi)
            indices_matrix = torch.stack(indices_matrix, dim=0) # (seq_len, P)
            particles_seq = torch.stack(particles_seq, dim=0)
            total_time = time.time() - start_time
            self.logger.info("PMS TIME - {} particles - {} seq len: {}".format(self.num_particles, self.seq_len, total_time))
            return (mses, errors), phis, (indices_matrix.numpy(), particles_seq.squeeze().numpy())

    def get_error(self, k):
        estimation = self.trajectories * self.filtering_weights.view(self.filtering_weights.shape[0], self.filtering_weights.shape[1],1,1) # element-wise multiplication of resampled trajectories and w_n (last filtering weights)
        estimation = estimation.sum(1) # (B,S,hidden_size) # weighted sum.
        error = estimation - self.states.squeeze(1)[:,:k+2,:]
        mse = np.square(error.numpy()).mean(-1) # shape (B,S) # square of error and mean over last dim (all dimensions of the hidden states).
        return error.squeeze(), mse.squeeze(), estimation.squeeze()

    def get_genealogy(self, indices_matrix):
        n_particles = indices_matrix.shape[-1]
        n_times = indices_matrix.shape[0]
        particle_indices = np.arange(n_particles, dtype=int)
        # Array contanant la genealogie
        # La genealogie est un array n_times * n_particles
        # A chaque ligne on a l'indice de particule en lequel passe la trajectoire
        # Au debut tout le monde passe par sa particule associÃ©e.
        genealogy = np.repeat([particle_indices], n_times+1, axis=0)
        # Maintenant on actualise
        for t in range(0, n_times):
            old_genealogy = genealogy  # A chaque debut, on stocke la "vieille genealogie"
            # Ici, pour l'exemple, un resampling uniforme
            indice_resampling = indices_matrix[t]
            # Maintenant, pour chaque colonne, la colonne entiere est remplacee par l'ancienne associÃ©e Ã  la particule
            genealogy = old_genealogy[:, indice_resampling]
            # Attention, Ã  chaque fois on restipule bien qu'au temps final, on passe par le bon indice de particule
            genealogy[t + 1:, :] = particle_indices
        return genealogy

    def resample_trajectories(self, trajectories, genealogy):
        n_particles = genealogy.shape[-1]
        n_times = trajectories.shape[0]
        num_dim = trajectories.shape[-1]
        resampled_trajectories = np.zeros(shape=(n_times, n_particles, num_dim))
        for t in reversed(range(n_times)):
            resampled_trajectories[t, :, :] = trajectories[t, genealogy[t, :], :] # (seq_len, P, hidden_size)
        return np.transpose(resampled_trajectories, axes=[1,0,2])
