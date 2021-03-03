'''
Implementation of a Stochastic RNN.
Inspired from: https://github.com/pytorch/pytorch/issues/11335
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from smc.resample import resample

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
        self.sigma_h = 0.5
        self.num_particles = 10

    def add_noise(self, params, sigma):
        '''
        :param params: tensor to which noise should be added.
        :param sigma: covariance matrix.
        :return:
        '''
        #gaussian_noise = torch.normal(mean=0.0, std=1.0, size=params.size, dtype=params.dtype)
        gaussian_noise = torch.normal(mean=params.new_zeros(params.size()), std=params.new_ones(params.size()))
        noise = (sigma) ** (1 / 2) * gaussian_noise
        return params + noise

    def forward(self, input, hidden=None):
        #self.check_forward_input(input)  # check correct shape of input for forward pass.
        if hidden is None:
            # initialization of h to zeros tensors with same dtype & device than input.
            hx = input.new_zeros(input.size(0), self.num_particles, self.hidden_size,
                                 requires_grad=False)  # hx should be of size (batch_size, hidden_size).
        else:
            hx = hidden

        #self.check_forward_hidden(input=input, hx=hx,
                                   #hidden_label='[0]')  # check correct shape of hidden compared to input for h.

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
        self.rnn = RNNCell(input_size, hidden_size, bias=bias, activation=activation)
        self.sigma_y = 0.5
        self.num_particles = 10
        self.batch_size = 16
        self.resampling_weights = F.softmax(torch.randn(self.batch_size, self.num_particles))

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)


    def compute_w_regression(self, predictions, targets):
        '''
        # FORMULA
        # logw = -0.5 * mu_t ^ T * mu_t / omega
        # logw = logw - max(logw)
        # w = exp(logw)
        :param predictions: output of final layer: (B,P,1,F_y)
        :param y: current target element > shape (B,P,1,F_y).
        :return:
        resampling weights of shape (B,P).
        '''
        mu_t = targets - predictions  # (B,P,1,F_y)
        mu_t = mu_t.squeeze(-2)  # removing sequence dim. # (B,P,F_y).
        log_w = (-1 / (2 * self.sigma_y)) * torch.matmul(mu_t, mu_t.permute(0,2,1))  # (B,P,P)
        log_w = torch.diagonal(log_w, dim1=-2, dim2=-1)  # take the diagonal. # (B,P).
        w = F.softmax(log_w)
        return w

    def forward(self, input, target, hidden=None):
        batch_size, num_particles, seq_len, hidden_size = input.size()
        if hidden is None:
            hx = input.new_zeros(batch_size, self.num_particles, self.hidden_size, requires_grad=False)
        else:
            hx = hidden
        ht = []
        for t in range(seq_len):
            x = input[:, :,  t, :]
            y = target[:, :,  t, :]
            # Mutation: compute $I_t$ from $w_{t-1}$ and resample $h_{t-1}
            It = torch.multinomial(self.resampling_weights, num_samples=self.num_particles) #TODO detach I
            resampled_h = resample(hx, It)
            hy = self.rnn(x, resampled_h)
            ht.append(hy)
            hx = hy
            # compute $w_t$ from $h_t$ and $Y_t$
            predt = self.fc(hy)
            self.resampling_weights = self.compute_w_regression(predictions=predt, targets=y) #TODO: detach resampling weights.
        output = torch.stack(ht,
                             dim=1)  # sequence of hidden states from last layer. shape (S,B,hidden_size).
        return output, hy



class StochasticRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_particles=10, num_layers=1, bias=True, activation='tanh'):
        super(StochasticRNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_particles = num_particles
        self.sigma_y = 0.5
        if num_layers == 1:
            self.rnn = OneLayerRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, bias=bias, activation=activation)
        elif num_layers > 1:
            self.rnn = StochasticRNN(input_size=input_size, hidden_size=hidden_size, bias=bias, activation=activation, num_layers=num_layers)
        self.fc = self.rnn.fc

    def forward(self, input, target):
        if input.size(1) == 1:
            input = input.repeat(1,self.num_particles, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, self.num_particles, 1, 1)
        output, hidden = self.rnn(input=input, target=target)  # output (B,S,P,hidden_size), hidden: (num_layers, B, hidden_size)
        logits = self.fc(output)  # (B,S,P,output_size)
        return logits



class StochasticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, p_drop=0., bias=True, activation='tanh', num_layers=1):
        super(StochasticRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden = nn.ModuleList([RNNCell(input_size=(input_size if layer == 0 else hidden_size),
                                              hidden_size=hidden_size,
                                              p_drop=(0. if layer == num_layers - 1 else p_drop), bias=bias, activation=activation) for layer in
                                     range(num_layers)]) # shape (num_layers, batch_size, hidden_size).


    def forward(self, input, hidden=None):
        batch_size, seq_len, hidden_size = input.size()
        if hidden is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx = hidden
        ht = []
        h = hx
        for t in range(seq_len):
            x = input[:, t, :]
            h_t_l = []
            for l, layer in enumerate(self.hidden):
                h_tl = layer(input=x, hidden=h[l])
                x = h_tl
                h_t_l.append(h_tl) # get all hidden states (of all layers) for one timestep.
            ht.append(h_t_l) # get all hidden states for all timesteps.
            h = ht[t] # current hidden state (of timestep t)
        output = torch.stack([h[-1] for h in ht],
                             dim=1)  # sequence of hidden states from last layer. shape (S,B,hidden_size).
        hy = torch.stack(ht[-1])  # last hidden state (of the last timestep). shape (num_layers, B, hidden_size).

        return output, hy

if __name__ == '__main__':
    input = torch.randn(16, 1, 24, 4)
    target = torch.randn(16, 1, 24, 4)
    rnn_network = StochasticRNNModel(input_size=input.size(-1), hidden_size=32, output_size=input.size(-1))
    output = rnn_network(input, target)
    print('output', output.shape)