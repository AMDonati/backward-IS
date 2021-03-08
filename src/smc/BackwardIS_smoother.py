import torch
from smc.BootstrapFilter import RNNBootstrapFilter
from smc.utils import resample

class RNNBackwardISSmoothing:
    def __init__(self, bootstrap_filter, observations, states, backward_samples, estimation_function):
        '''
        :param bootstrap_filter:
        :param observations:
        :param states:
        :param backward_samples:
        :param estimation_function:
        '''
        self.bootstrap_filter = bootstrap_filter
        self.rnn = bootstrap_filter.rnn
        self.observations = observations # Tensor of shape (B, particles, seq_len, output_size) # Add batch_size as well?
        self.states = states # Tensor of shape (B, particles, seq_len, hidden_size)
        self.backward_samples = backward_samples
        self.num_particles = self.bootstrap_filter.num_particles
        self.estimation_function = estimation_function
        self.seq_len = self.observations.size(-2)

        self.init_particles()

    def init_particles(self):
        self.ancestors = self.states[:,:,0,:] # (B, num_particles, hidden_size)
        preds = self.rnn.fc(self.ancestors) #TODO: put this inside the bootstrap filter instead.
        self.filtering_weights = self.bootstrap_filter.compute_filtering_weights(predictions=preds, targets=self.observations[:,:,0,:]) #decide if take $Y_0 of $Y_1$
        self.past_tau = torch.zeros(self.states.size(0), self.num_particles, self.states.size(-1))
        self.new_tau = self.past_tau

    def compute_IS_weights(self, ancestor, particle):
        '''return the importance sampling weights \Tilde(w)(i,j)'''

    def update_tau(self, ancestors, particle, backward_indices, IS_weights, k):
        #'''update $\tau_k^l from $\tau_{k-1}^l, $w_{k-1]^l, $\xi_{k-1}^Jk$ for all Jk(j), \Tilde(w)(l,j) for all j in 0...backward samples'''
        sum_weights = IS_weights.sum(1) # shape (B)
        resampled_tau = resample(self.past_tau, backward_indices) # (B,backward_samples, hidden_size)
        #resampled_states = resample(self.states[:,:,k,:], backward_indices) #k or (k-1) ? Or do we use the ancestors ? I think so...
        new_tau_element = IS_weights * (resampled_tau + self.estimation_function(k, ancestors)) # (B, backward_samples, hidden_size) # are we doing it like that ?
        new_tau = new_tau_element.sum(1) / sum_weights
        return new_tau

    def estimate_conditional_expectation_of_function(self):
        #TODO: add a with torch.no_grad()
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
                    ancestor = resample(self.ancestors, backward_indice) # shape (B, 1, hidden) #TODO: check function for one single indice.
                    selected_ancestors.append(ancestor)
                    # C. Compute IS weights with Ancestor & Particle.
                    is_weight = self.rnn.estimate_transition_density(ancestor=ancestor, particle=particle)
                    IS_weights.append(is_weight)
                # End for
                selected_ancestors = torch.stack(selected_ancestors, dim=1).squeeze() # dim (B, backward_samples, 1, hidden_size)
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