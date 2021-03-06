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
        # ADDING NOISE.
        activation = self.add_noise(activation, self.sigma_h)
        if self.activation_fn == 'tanh':
            hy = activation.tanh()
        elif self.activation_fn == 'relu':
            hy = activation.relu()
        return hy


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

    def generate_observations(self, initial_input, seq_len, sigma_init, sigma_h, sigma_y, num_samples=100):
        self.update_sigmas(sigma_h=sigma_h, sigma_y=sigma_y, sigma_init=sigma_init)
        with torch.no_grad():
            initial_input = initial_input.repeat(1, num_samples, 1, 1)
            input = initial_input
            initial_hidden = input.new_zeros(input.size(0), num_samples, self.hidden_size, requires_grad=False)
            initial_hidden = self.rnn_cell.add_noise(initial_hidden, self.sigma_init)
            for k in range(seq_len-1):
                observations, (hidden, _) = self.forward(input=input, hidden=initial_hidden)
                input = torch.cat([input, observations[:,:,-1,:].unsqueeze(dim=-2)], dim=2)
            hidden = torch.cat([initial_hidden.unsqueeze(dim=-2), hidden], dim=-2) # adding initial hidden_state.
        return input, hidden

    def gaussian_density_function(self, X, mean, covariance):
        #distrib = torch.distributions.normal.Normal(loc=mean, scale=covariance**(1/2))
        #dd = distrib.cdf(X)
        #distrib_2 = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance * torch.eye(mean.size(-1)))
        #dd_2 = distrib_2.cdf(X)
        mu = X - mean   # (B,P,H)
        density = torch.exp((-1 / (2 * covariance)) * torch.matmul(mu, mu.permute(0, 2, 1)))  # (B,P,P)
        density = torch.diagonal(density, dim1=-2, dim2=-1)  # take the diagonal. # (B,P).
        return density

    def estimate_transition_density(self, particle, ancestor):
        # compute gaussian density of inv_tanh(new_particle)
        d = self.gaussian_density_function(inv_tanh(particle), ancestor, self.rnn_cell.sigma_h)
        # compute prods of 1 / derive_tanh(inv_tanh(new_particle))
        transform = derive_tanh(inv_tanh(particle)) # (B, P, hidden)
        inv_transform = torch.pow(transform, -1)
        prod = inv_transform.prod(dim=-1) # (B,P)
        w = d * prod # (B,P)
        w = F.softmax(w)
        return w

    def forward(self, input, hidden=None):
        batch_size, num_particles, seq_len, hidden_size = input.size()
        if hidden is None:
            hx = input.new_zeros(batch_size, num_particles, self.hidden_size, requires_grad=False)
        else:
            hx = hidden
        ht = []
        for t in range(seq_len):
            x = input[:, :,  t, :]
            hy = self.rnn_cell(x, hx)
            ht.append(hy)
            hx = hy
        hidden = torch.stack(ht, dim=1)
        logits = self.fc(hidden)
        observations = self.rnn_cell.add_noise(logits, self.sigma_y) # (B,S,P,F_y)
        observations = observations.permute(0,2,1,3)
        hidden = hidden.permute(0,2,1,3)
        return observations, (hidden, hy)


if __name__ == '__main__':
    input = torch.randn(16, 1, 24, 4)
    target = torch.randn(16, 1, 24, 4)
    rnn_network = OneLayerRNN(input_size=input.size(-1), hidden_size=32, output_size=input.size(-1))
    logits, output, hidden = rnn_network(input, target)
    print('output', output.shape)