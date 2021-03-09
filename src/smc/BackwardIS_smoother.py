import torch
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.utils import resample
import torch.nn as nn

class RNNBackwardISSmoothing:
    def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function):
        '''
        :param bootstrap_filter: Class Implementing a Bootstrap filter algorithm.
        :param observations: sequence of observations generated by a stochastic RNN: tensor of shape (num_samples=B, num_particles, seq_len, output_size)
        :param states: sequence of hidden states generated by a stochastic RNN: tensor of shape (num_samples, num_particles, seq_len, hidden_size)
        :param backward_samples: number of backward_samples for the Backward IS Smoothing algo.
        :param estimation_function: Fonction to estimate: in our case $mathbb[E][X_0|Y_{0:n}]$
        '''
        self.bootstrap_filter = bootstrap_filter
        self.rnn = bootstrap_filter.rnn
        self.observations = observations # Tensor of shape (B, particles, seq_len, output_size)
        self.states = states # Tensor of shape (B, particles, seq_len, hidden_size)
        self.backward_samples = backward_samples
        self.num_particles = self.bootstrap_filter.num_particles
        self.estimation_function = estimation_function
        self.seq_len = self.observations.size(-2)

        self.init_particles()

    def init_particles(self):
        self.ancestors = self.states[:,:,0,:] # (B, num_particles, hidden_size)
        self.filtering_weights = self.bootstrap_filter.compute_filtering_weights(hidden=self.ancestors, observations=self.observations[:,:,0,:]) #decide if take $Y_0 of $Y_1$
        self.past_tau = torch.zeros(self.states.size(0), self.num_particles, self.states.size(-1))
        self.new_tau = self.past_tau


    def update_tau(self, ancestors, particle, backward_indices, IS_weights, k):
        '''
        :param ancestors: ancestors particles $\xi_{k-1}^Jk$ sampled with backward_indices. tensor of shape (B, backward_samples, hidden_size)
        :param particle: $\xi_k^l$ (here not used in the formula of the estimation function): tensor of shape (B, 1, hidden_size)
        :param backward_indices: $J_k(j)$: tensor of shape (B, backward_samples)
        :param IS_weights: normalized importance sampling weights: tensor of shape (B, backward_samples)
        :param k: current timestep.
        '''
        #'''update $\tau_k^l from $\tau_{k-1}^l, $w_{k-1]^l, $\xi_{k-1}^Jk$ and from Jk(j), \Tilde(w)(l,j) for all j in 0...backward samples'''

        resampled_tau = resample(self.past_tau, backward_indices) # (B,backward_samples, hidden_size)
        new_tau_element = IS_weights * (resampled_tau + self.estimation_function(k, ancestors)) # (B, backward_samples, hidden_size) #TODO: check that it is the correct formula.
        new_tau = new_tau_element.sum(1)
        return new_tau

    def estimate_conditional_expectation_of_function(self):
        with torch.no_grad():
            # for loop on time
            for k in range(self.seq_len - 1):
                # Run bootstrap filter at time k
                self.old_filtering_weights = self.filtering_weights
                self.past_tau = self.new_tau
                self.particles, self.filtering_weights = self.bootstrap_filter.get_new_particle(
                    observation=self.observations[:, :, k, :], next_observation=self.observations[:, :, k + 1, :],
                    hidden=self.ancestors, weights=self.old_filtering_weights)
                # Backward Simulation
                # For loop of number of particles
                new_taus = []
                for l in range(self.num_particles):
                    # Select one particle.
                    particle = self.particles[:, l, :].unsqueeze(dim=1) # shape (B, 1, hidden)
                    selected_ancestors = []
                    selected_backward_indices = []
                    IS_weights = []
                    # For loop on Backward Samples
                    for j in range(self.backward_samples):
                        # A. Get backward Indice J from past filtering weights
                        backward_indice = torch.multinomial(self.old_filtering_weights, 1) # shape (B, 1)
                        selected_backward_indices.append(backward_indice)
                        # B. Select Ancestor with J.
                        ancestor = resample(self.ancestors, backward_indice) # shape (B, 1, hidden)
                        selected_ancestors.append(ancestor)
                        # C. Compute IS weights with Ancestor & Particle.
                        is_weight = self.rnn.estimate_transition_density(ancestor=ancestor, particle=particle)
                        IS_weights.append(is_weight)
                    # End for
                    selected_ancestors = torch.stack(selected_ancestors, dim=1).squeeze() # dim (B, backward_samples, hidden_size)
                    selected_indices = torch.stack(selected_backward_indices, dim=1).squeeze() # dim (B, backward_samples)
                    IS_weights = torch.stack(IS_weights, dim=1) # dim (backward samples, B, 1)
                    # compute $\tau_k^l$ with all backward IS weights, ancestors, current particle & all backward_indices.
                    new_tau = self.update_tau(ancestors=selected_ancestors, particle=particle, backward_indices=selected_indices, IS_weights=IS_weights, k=k)
                    new_taus.append(new_tau)
                # End for
                self.new_tau = torch.stack(new_taus, dim=1) # shape (B, num_particles, hidden_size)
            # End for
            # Compute $\phi_n$ with last filtering weights and last $tau$.
            phi_element = self.filtering_weights.unsqueeze(-1) * self.new_tau
            phi = phi_element.sum(1) # shape (B, hidden_size)
        return phi

    def compute_mse_phi_X0(self, phi):
        ''':param phi: estimation of $mathbb[E][X_0|Y_{0:n}]$: tensor of shape (B, hidden size).
        '''
        criterion = nn.MSELoss()
        loss = criterion(phi, self.states[:,:,0,:].mean(dim=1))
        return loss