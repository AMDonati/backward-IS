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

    def compute_filtering_weights(self, predictions, targets):
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
        mu_t = targets - predictions  # (B,P,F_y)
        log_w = (-1 / (2 * self.rnn.sigma_y)) * torch.matmul(mu_t, mu_t.permute(0, 2, 1))  # (B,P,P)
        log_w = torch.diagonal(log_w, dim1=-2, dim2=-1)  # take the diagonal. # (B,P).
        w = F.softmax(log_w)
        return w


    def get_new_particle(self, observation, next_observation, hidden, weights):
        # Mutation: compute $I_t$ from $w_{t-1}$ and resample $h_{t-1}
        It = torch.multinomial(weights, num_samples=self.num_particles)
        resampled_h = resample(hidden, It)
        # Selection : get $h_t$
        new_hidden = self.rnn_cell(observation, resampled_h)
        predictions = self.rnn.fc(new_hidden)
        # compute $w_t$
        new_weights = self.compute_filtering_weights(predictions=predictions, targets=next_observation)
        return new_hidden, new_weights