from smc.utils import resample, resample_1D, log_gaussian_density_function, manual_log_density_function
import torch
import torch.nn.functional as F
import numpy as np

class RNNBootstrapFilter:
    def __init__(self, num_particles, rnn):
        self.num_particles = num_particles
        self.rnn = rnn
        self.rnn_cell = self.rnn.rnn_cell

    def init_filtering_weights(self):
        pass

    def compute_filtering_weights(self, hidden, observations):
        '''
             # FORMULA
             # logw = -0.5 * mu_t ^ T * mu_t / sigma; sigma=scalar covariance.
             #  w = softmax(log_w)
             :param hidden: hidden state at timestep k: tensor of shape (B,num_particles,1,hidden_size)
             :param observations: current target element > shape (B,num_particles,1,F_y).
             :return:
             resampling weights of shape (B,P=num_particles).
        '''
        # get current prediction from hidden state.
        predictions = self.rnn.fc(hidden) # (B, P, F_y)
        observations = observations.repeat(1, self.num_particles, 1)
        log_w = self.rnn.log_gaussian_density_function(X=observations, mean=predictions, covariance=self.rnn.sigma_y)
        w = F.softmax(log_w)
        return w

    def get_new_particle(self, observation, next_observation, hidden, weights, resampling=True):
        #'''
        #:param observation: current observation $Y_{k}$: tensor of shape (B, 1, input_size)
        #:param next_observation $Y_{k+1}$: tensor of shape (B, P, input_size)
        #:param hidden $\xi_k^l(h_k)$: current hidden state: tensor of shape (B, P, hidden_size)
        #:param $\w_{k-1}^l$: previous resampling weights: tensor of shape (B, P)
        #:return new_hidden state $\xi_{k+1}^l$: tensor of shape (B, P, hidden_size), new_weights $w_k^l$: shape (B,P).
        #'''
        if resampling:
            # Mutation: compute $I_t$ from $w_{t-1}$ and resample $h_{t-1}$ = \xi_{t-1}^l
            It = torch.multinomial(weights, num_samples=self.num_particles, replacement=True)
            resampled_h = resample(hidden, It)
        else:
            resampled_h = hidden
        # Selection : get $h_t$ = \xi_t^l
        observation = observation.repeat(1, self.num_particles, 1)
        new_hidden, _ = self.rnn_cell(observation, resampled_h)
        # compute $w_t$
        new_weights = self.compute_filtering_weights(hidden=new_hidden, observations=next_observation)
        return (new_hidden, resampled_h), new_weights

class SVBootstrapFilter:
    def __init__(self, num_particles, init_params):
        self.num_particles = num_particles
        self.params = init_params

    def update_SV_params(self, params):
        self.params = params

    def compute_filtering_weights(self, particle, observation, params):
        '''
             # FORMULA
             # logw = -0.5 * mu_t ^ T * mu_t / sigma; sigma=scalar covariance.
             #  w = softmax(log_w)
             :param hidden: hidden state at timestep k: tensor of shape (B,num_particles,1,hidden_size)
             :param observations: current target element > shape (B,num_particles,1,F_y).
             :return:
             resampling weights of shape (B,P=num_particles).
        '''
        # get current prediction from hidden state.
        observation = observation.unsqueeze(-1)
        covariance_diag = torch.exp(particle).unsqueeze(-1) # shape (P,1,1)
        log_w = log_gaussian_density_function(X=observation, mean=torch.zeros(size=observation.size()), covariance=torch.exp(params[2]) * covariance_diag)  # shape (P)
        #log_w2 = manual_log_density_function(X=observation, mean=torch.zeros(size=observation.size()), covariance=self.params[2]**2 * covariance_diag)
        w = F.softmax(log_w, dim=-1)
        return w

    def compute_IS_weights(self, resampled_ancestors, particle, next_observation, backward_samples, params):
        particle = particle.unsqueeze(1).repeat((1, backward_samples,1)) # shape (particles, backward_samples, 1)
        # transition density
        log_w = log_gaussian_density_function(X=particle, mean=params[0]*resampled_ancestors, covariance=torch.exp(params[1]))
        log_w2 = manual_log_density_function(X=particle, mean=params[0]*resampled_ancestors, covariance=torch.exp(params[1]) * torch.eye(1))
        # observation density
        log_w_o = log_gaussian_density_function(X=next_observation, mean=torch.zeros(next_observation.size()), covariance=torch.exp(params[2]) * torch.exp(particle).unsqueeze(-1))
        w = F.softmax(log_w + log_w_o, dim=-1)
        return w

    def get_new_particle(self, next_observation, ancestor, weights, params, resampling=True):
        #'''
        #:param next_observation $Y_{k+1}$: tensor of shape (B, P, input_size)
        #:param ancestor \xi_k:  tensor of shape (B, P, hidden_size)
        #:param $\w_{k}^l$: previous resampling weights: tensor of shape (B, P)
        #:return new_hidden state $\xi_{k+1}^l$: tensor of shape (B, P, hidden_size), new_weights $w_{k+1}^l$: shape (B,P).
        #'''
        if resampling:
            # Mutation: compute $I_t$ from $w_{t-1}$ and resample $h_{t-1}$ = \xi_{t-1}^l
            It = torch.multinomial(weights, num_samples=self.num_particles, replacement=True)
            resampled_ancestor = resample_1D(ancestor, It) # OK function resample works. # shape (P,1)
        else:
            resampled_ancestor = ancestor
        # Selection : get $h_t$ = \xi_t^l
        particle = params[0] * resampled_ancestor + torch.exp(params[1]/2) * torch.normal(mean=ancestor.new_zeros(ancestor.size()), std=ancestor.new_ones(ancestor.size()))
        # compute $w_t$
        new_weights = self.compute_filtering_weights(particle=particle, observation=next_observation, params=params)
        return particle, new_weights