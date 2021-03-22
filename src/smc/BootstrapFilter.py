from smc.utils import resample
import torch
import torch.nn.functional as F

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
        #mu_t = observations - predictions  # (B,P,F_y)
        #log_w = (-1 / (2 * self.rnn.sigma_y)) * torch.matmul(mu_t, mu_t.permute(0, 2, 1))  # (B,P,P)
        # = torch.diagonal(log_w, dim1=-2, dim2=-1)  # take the diagonal. # (B,P).
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
            It = torch.multinomial(weights, num_samples=self.num_particles)
            resampled_h = resample(hidden, It)
        else:
            resampled_h = hidden
        # Selection : get $h_t$ = \xi_t^l
        observation = observation.repeat(1, self.num_particles, 1)
        new_hidden, _ = self.rnn_cell(observation, resampled_h)
        # compute $w_t$
        new_weights = self.compute_filtering_weights(hidden=new_hidden, observations=next_observation)
        return (new_hidden, resampled_h), new_weights