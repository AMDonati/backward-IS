'''
Implementation of a Stochastic RNN.
Inspired from: https://github.com/pytorch/pytorch/issues/11335
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def update_sigmas(self, sigma_h, sigma_y):
        '''To inject noise after training.'''
        self.sigma_y = sigma_y
        self.rnn_cell.sigma_h = sigma_h

    def estimate_transition_density(self, new_h, old_h):
        #TODO: to update this function with right_formula.
        mu_t = new_h - old_h  # (B,P,1,F_y)
        mu_t = mu_t.squeeze(-2)  # removing sequence dim. # (B,P,F_y).
        log_w = (-1 / (2 * self.rnn_cell.sigma_h)) * torch.matmul(mu_t, mu_t.permute(0, 2, 1))  # (B,P,P)
        log_w = torch.diagonal(log_w, dim1=-2, dim2=-1)  # take the diagonal. # (B,P).
        w = F.softmax(log_w)
        return w

    def forward(self, input, hidden=None):
        batch_size, num_particles, seq_len, hidden_size = input.size()
        if hidden is None:
            hx = input.new_zeros(batch_size, num_particles, self.hidden_size, requires_grad=False) #TODO: add noise here also and store h0.
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
        observations = self.rnn_cell.add_noise(logits, self.sigma_y)
        return observations, (hidden, hy)


if __name__ == '__main__':
    input = torch.randn(16, 1, 24, 4)
    target = torch.randn(16, 1, 24, 4)
    rnn_network = OneLayerRNN(input_size=input.size(-1), hidden_size=32, output_size=input.size(-1))
    logits, output, hidden = rnn_network(input, target)
    print('output', output.shape)