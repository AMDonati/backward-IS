'''
Implementation of a Stochastic RNN.
Inspired from: https://github.com/pytorch/pytorch/issues/11335
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from smc.utils import inv_tanh, derive_tanh
import torch.distributions as distrib

class RNNCell(nn.RNNCell):
    '''
       ~RNNCell.weight_ih – the learnable input-hidden weights, of shape (hidden_size, input_size)
       ~RNNCell.weight_hh – the learnable hidden-hidden weights, of shape (hidden_size, hidden_size)
       ~RNNCell.bias_ih – the learnable input-hidden bias, of shape (hidden_size)
       ~RNNCell.bias_hh – the learnable hidden-hidden bias, of shape (hidden_size)
       '''

    def __init__(self, input_size, hidden_size, bias=True, activation='tanh'):
        super(RNNCell, self).__init__(input_size=input_size, hidden_size=hidden_size, bias=bias)
        self.activation_fn = activation
        self.sigma_h = 0

    def add_noise(self, params, sigma):
        '''
        :param params: tensor to which noise should be added.
        :param sigma: covariance matrix.
        :return:
        '''
        gaussian_noise = torch.normal(mean=params.new_zeros(params.size()), std=params.new_ones(params.size()))
        noise = (sigma) ** (1 / 2) * gaussian_noise
        return params + noise

    def forward(self, input, hidden=None):
        '''forward pass of the RNN Cell computing the next hidden state $h_k$ from $h_{k-1}$ and input $Y_k$'''
        if hidden is None:
            # initialization of h to zeros tensors with same dtype & device than input.
            batch_size, num_particles, seq_len, input_size = input.size()
            hx = input.new_zeros(batch_size, num_particles, self.hidden_size,
                                 requires_grad=False)  # hx should be of size (batch_size, num_particles, hidden_size).
        else:
            hx = hidden

        # computation of activation:
        activation = F.linear(input=input, weight=self.weight_ih, bias=self.bias_ih) + F.linear(input=hx,
                                                                                        weight=self.weight_hh,
                                                                                        bias=self.bias_hh)  # shape (batch_size, hidden_size)
        # adding noise for the transition model P(h_{k+1} | h_k)
        activation_ = self.add_noise(activation, self.sigma_h)
        if self.activation_fn == 'tanh':
            hy = activation_.tanh()
        elif self.activation_fn == 'relu':
            hy = activation_.relu()
        return hy, activation


class OneLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=True, activation='tanh'):
        super(OneLayerRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(input_size, hidden_size, bias=bias, activation=activation)
        self.sigma_y = 0
        self.sigma_init = 0
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def update_sigmas(self, sigma_h, sigma_y, sigma_init):
        '''To inject noise after training.'''
        self.sigma_y = sigma_y
        self.rnn_cell.sigma_h = sigma_h
        self.sigma_init = sigma_init

    def generate_observations(self, initial_input, seq_len, sigma_init, sigma_h, sigma_y):
        '''
        :param initial_input: Y_0: tensor for shape (num data samples = B, 1, 1, input_size)
        :param seq_len: length of the sequence of observations to generate.
        :param sigma_init: $\sigma$: covariance matrix for the initial hidden state $X_0(h_0)$
        :param sigma_h: $\nu_k$: scalar variance value of gaussian noise for the hidden state
        :param sigma_y: $\epsilon_k$: scalar variance value of gaussian noise for the observation model.
        :param num_samples: number of samples to generate.
        '''
        # Update variance values
        self.update_sigmas(sigma_h=sigma_h, sigma_y=sigma_y, sigma_init=sigma_init)
        with torch.no_grad():
            input = initial_input
            # initialize X_0 with gaussian noise of variance $sigma_init$.
            initial_hidden = input.new_zeros(input.size(0), 1, self.hidden_size, requires_grad=False)
            initial_hidden = self.rnn_cell.add_noise(initial_hidden, self.sigma_init)
            for k in range(seq_len-1):
                observations, (hidden, _) = self.forward(input=input, hidden=initial_hidden)
                input = torch.cat([input, observations[:,:,-1,:].unsqueeze(dim=-2)], dim=2)
            hidden = torch.cat([initial_hidden.unsqueeze(dim=-2), hidden], dim=-2) # adding initial hidden_state.
        return input, hidden

    def log_gaussian_density_function(self, X, mean, covariance):
        ''' Compute the Gaussian Density Function with mean, diagonal covariance diag(covariance) at input X.
        :param X: tensor of shape (B, P, hidden_size)
        :param mean: tensor of shape (B, P, hidden_size)
        :param covariance: scalar value.
        '''
        distrib = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance * torch.eye(mean.size(-1)))
        log_d = distrib.log_prob(X)
        return log_d

    def estimate_transition_density(self, particle, ancestor, previous_observation):
        #'''
            #Compute the transition density function $q_k(X_{k+1}|X_k)$ for X_{K+1) = particle, $X_k$ = ancestor and Y_k = previous_observation.
            #:param particle $\xi_{k-1)$: shape (B, 1, hidden_size)
            #:param ancestor $\xi_{k}^J$: shape (B, J, hidden_size)
            #:param previous observation $Y_{k}^J$: shape (B, 1, input_size)
        #'''
        # compute mean of gaussian density function from ancestor: $\mu = W_1 * ancestor + W_2 * prev_observation + b$
        previous_observation = previous_observation.repeat(ancestor.size(0), ancestor.size(1), 1)
        _, activation = self.rnn_cell(input=previous_observation, hidden=ancestor) # shape (B, J, hidden_size)
        # compute gaussian density of arctanh(new_particle)
        log_density = self.log_gaussian_density_function(X=torch.atanh(particle).unsqueeze(1), mean=activation, covariance=self.rnn_cell.sigma_h)
        # compute prods of 1 / derive_tanh(inv_tanh(new_particle))
        transform = (1-torch.pow(particle, 2)) # (1-z^2) element-wise. (B, 1, hidden)
        log_inv_transform = torch.pow(transform, -1).log() # 1 / (1-z^2) element-wise.
        sum = log_inv_transform.sum(dim=-1) # (B,1) # sum_{i} 1 / (1-z_i^2)
        log_w = log_density + sum.unsqueeze(-1) # (B,J)
        # normalize weights with a softmax.
        w = F.softmax(log_w, dim=-1)
        return w

    def forward(self, input, hidden=None):
        '''forward pass for the RNN'''
        batch_size, num_particles, seq_len, hidden_size = input.size()
        if hidden is None:
            hx = input.new_zeros(batch_size, num_particles, self.hidden_size, requires_grad=False)
        else:
            hx = hidden
        ht = []
        for t in range(seq_len):
            x = input[:, :,  t, :]
            hy, _ = self.rnn_cell(x, hx)
            ht.append(hy)
            hx = hy
        hidden = torch.stack(ht, dim=1)
        logits = self.fc(hidden)
        # adding noise for the observation model P(Y_{k+1} | h_k)
        observations = self.rnn_cell.add_noise(logits, self.sigma_y) # (B,S,P,F_y)
        observations = observations.permute(0,2,1,3)
        hidden = hidden.permute(0,2,1,3) # shape (B,P,S,F_y)
        return observations, (hidden, hy)


if __name__ == '__main__':
    input = torch.randn(16, 1, 24, 4)
    target = torch.randn(16, 1, 24, 4)
    rnn_network = OneLayerRNN(input_size=input.size(-1), hidden_size=32, output_size=input.size(-1))
    logits, output, hidden = rnn_network(input, target)
    print('output', output.shape)