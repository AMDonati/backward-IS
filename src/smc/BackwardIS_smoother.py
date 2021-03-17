import torch
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.utils import resample, resample_all_seq
import torch.nn as nn
import os
import numpy as np

class RNNBackwardISSmoothing:
    def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function, save_elements=False):
        # '''
        # :param bootstrap_filter: Class Implementing a Bootstrap filter algorithm.
        # :param observations: sequence of observations generated by a stochastic RNN: tensor of shape (num_samples=B, num_particles, seq_len, output_size)
        # :param states: sequence of hidden states generated by a stochastic RNN: tensor of shape (num_samples, num_particles, seq_len, hidden_size)
        # :param backward_samples: number of backward_samples for the Backward IS Smoothing algo.
        # :param estimation_function: Fonction to estimate: in our case $mathbb[E][X_0|Y_{0:n}]$
        # '''
        self.bootstrap_filter = bootstrap_filter
        self.rnn = bootstrap_filter.rnn
        self.observations = observations # Tensor of shape (B, particles, seq_len, output_size)
        self.states = states # Tensor of shape (B, particles, seq_len, hidden_size)
        self.backward_samples = backward_samples
        self.num_particles = self.bootstrap_filter.num_particles
        self.estimation_function = estimation_function
        self.seq_len = self.observations.size(-2)
        self.save_elements = save_elements

        self.init_particles()

        self.taus = []
        self.all_IS_weights = []

    def init_particles(self):
        self.ancestors = self.states[:,:,0,:].repeat(1, self.num_particles, 1) # (B, num_particles, hidden_size)
        self.trajectories = self.ancestors.unsqueeze(-2)
        self.filtering_weights = self.bootstrap_filter.compute_filtering_weights(hidden=self.ancestors, observations=self.observations[:,:,0,:]) #decide if take $Y_0 of $Y_1$
        self.past_tau = torch.zeros(self.states.size(0), self.num_particles, self.states.size(-1))
        self.new_tau = self.past_tau
        if self.save_elements:
            self.taus.append(self.past_tau)


    def update_tau(self, ancestors, particle, backward_indices, IS_weights, k):
        # '''
        # :param ancestors: ancestors particles $\xi_{k-1}^Jk$ sampled with backward_indices. tensor of shape (B, backward_samples, hidden_size)
        # :param particle: $\xi_k^l$ (here not used in the formula of the estimation function): tensor of shape (B, 1, hidden_size)
        # :param backward_indices: $J_k(j)$: tensor of shape (B, backward_samples)
        # :param IS_weights: normalized importance sampling weights: tensor of shape (B, backward_samples)
        # :param k: current timestep.
        # '''
        #'''update $\tau_k^l from $\tau_{k-1}^l, $w_{k-1]^l, $\xi_{k-1}^Jk$ and from Jk(j), \Tilde(w)(l,j) for all j in 0...backward samples'''

        resampled_tau = resample(self.past_tau, backward_indices) # (B,backward_samples, hidden_size)
        new_tau_element = IS_weights * (resampled_tau + self.estimation_function(k, ancestors)) # (B, backward_samples, hidden_size)
        new_tau = new_tau_element.sum(1)
        return new_tau

    def estimate_conditional_expectation_of_function(self):
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
                # For loop of number of particles
                new_taus, all_is_weights = [], []
                for l in range(self.num_particles):
                    # Select one particle.
                    particle = self.particles[:, l, :].unsqueeze(dim=1) # shape (B, 1, hidden)
                    # A. Get backward Indice J from past filtering weights
                    backward_indices = torch.multinomial(self.old_filtering_weights, self.backward_samples) # shape (B, J)
                    # B. Select Ancestor with J.
                    ancestors = resample(self.ancestors, backward_indices) # shape (B, J, hidden)
                    # C. Compute IS weights with Ancestor & Particle.
                    is_weights = self.rnn.estimate_transition_density(ancestor=ancestors, particle=particle, previous_observation=self.observations[:,:,k,:])
                    # End for
                    # compute $\tau_k^l$ with all backward IS weights, ancestors, current particle & all backward_indices.
                    new_tau = self.update_tau(ancestors=ancestors, particle=particle, backward_indices=backward_indices, IS_weights=is_weights.unsqueeze(-1), k=k)
                    new_taus.append(new_tau)
                    all_is_weights.append(is_weights)
                # End for
                self.new_tau = torch.stack(new_taus, dim=1) # shape (B, num_particles, hidden_size)
                self.ancestors = self.particles
                if self.save_elements:
                    self.taus.append(self.new_tau)
                    self.all_IS_weights.append(torch.stack(all_is_weights, dim=1))
            # End for
            # Compute $\phi_n$ with last filtering weights and last $tau$.
            phi_element = self.filtering_weights.unsqueeze(-1) * self.new_tau
            phi = phi_element.sum(1) # shape (B, hidden_size)
        return phi


    def poor_man_smoother_estimation(self):
        with torch.no_grad():
            # for loop on time
            for k in range(self.seq_len - 1):
                # Selection: resample all past trajectories with current indice i_t
                self.old_filtering_weights = self.filtering_weights
                i_t = torch.multinomial(self.old_filtering_weights, num_samples=self.num_particles)
                resampled_trajectories = resample_all_seq(self.trajectories, i_t=i_t)
                ancestor = resampled_trajectories[:,:,k,:] # get resampled ancestor $\xi_{k-1}$
                # Mutation: Run bootstrap filter at time k to get new particle without resampling
                (self.particles, _), self.filtering_weights = self.bootstrap_filter.get_new_particle(
                    observation=self.observations[:, :, k, :], next_observation=self.observations[:, :, k + 1, :],
                    hidden=ancestor, weights=self.old_filtering_weights, resampling=False)
                # append resampled trajectories to new particle
                self.trajectories = torch.cat([resampled_trajectories, self.particles.unsqueeze(-2)], dim=-2)
            phi_element = self.filtering_weights.unsqueeze(-1) * self.estimation_function(X=self.trajectories[:,:,0,:], k=0) # shape (B,P,hidden_size)
            phi = phi_element.sum(1) # (B, hidden)
            return phi

    def debug_elements(self, data_path):
        if len(self.taus) > 0:
            taus = torch.stack(self.taus, dim=2).cpu().numpy()
            np.save(os.path.join(data_path, "taus.npy"), taus)
        if len(self.all_IS_weights) > 0:
            all_is_weights = torch.stack(self.all_IS_weights, dim=2).cpu().numpy()
            np.save(os.path.join(data_path, "IS_weights.npy"), all_is_weights)


    def compute_mse_phi_X0(self, phi):
        # ''':param phi: estimation of $mathbb[E][X_0|Y_{0:n}]$: tensor of shape (B, hidden size).
        # '''
        criterion = nn.MSELoss()
        loss = criterion(phi, self.states[:,:,0,:].mean(dim=1))
        return loss