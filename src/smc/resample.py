#import tensorflow as tf
import numpy as np
import torch

def resample(params, I):
    '''
    :params: shape (B,P,H)
    :I: shape (B,P)
    '''
    I = I.unsqueeze(-1).repeat(1,1,params.size(-1))
    resampled_params = torch.gather(input=params, index=I, dim=1)
    return resampled_params



if __name__ == "__main__":

    # ---------- test of corrected resample function -----------------------------------------------------------------------------------------------------------
    B = 2
    S = 3
    P = 4
    D = 1

    ind_matrix = tf.constant([[[1, 1, 2, 2], [0, 0, 0, 0], [1, 1, 1, 0]],
                              [[0, 1, 3, 2], [3, 3, 2, 0], [1, 2, 3, 1]]], shape=(B, S, P))
    ind_matrix = tf.transpose(ind_matrix, perm=[0, 2, 1])
    # ind_matrix = tf.tile(tf.expand_dims(ind_matrix, axis=0), multiples=[B, 1, 1])  # (B,P,S)

    print('indices_matrices', ind_matrix[0, :, :].numpy())

    K = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], shape=(B, S, P))
    K = tf.transpose(K, perm=[0, 2, 1])
    K = tf.expand_dims(K, axis=-1)  # (B,P,S,D=1)
    print('init K', K[0, :, :, 0])

    truth_t0_b1 = tf.constant([[2, 5, 9], [2, 6, 10], [3, 7, 11], [3, 8, 12]], shape=(P, S))
    truth_t1_b1 = tf.constant([[2, 5, 9], [2, 5, 10], [2, 5, 11], [2, 5, 12]], shape=(P, S))
    truth_t2_b1 = tf.constant([[2, 5, 10], [2, 5, 10], [2, 5, 10], [2, 5, 9]], shape=(P, S))
    truth_t0_b2 = tf.constant([[13, 17, 21], [14, 18, 22], [16, 19, 23], [15, 20, 24]], shape=(P, S))
    truth_t1_b2 = tf.constant([[15, 20, 21], [15, 20, 22], [16, 19, 23], [13, 17, 24]], shape=(P, S))
    truth_t2_b2 = tf.constant([[15, 20, 22], [16, 19, 23], [13, 17, 24], [15, 20, 22]], shape=(P, S))

    truth_t0 = tf.stack([truth_t0_b1, truth_t0_b2], axis=0)
    truth_t1 = tf.stack([truth_t1_b1, truth_t1_b2], axis=0)
    truth_t2 = tf.stack([truth_t2_b1, truth_t2_b2], axis=0)

    new_K = K
    for t in range(S):
        i_t = ind_matrix[:, :, t]
        new_K = resample(params=new_K, i_t=i_t, t=t)
        print('new K at time_step for batch 0 {}: {}'.format(t, new_K[0, :, :, 0]))
        print('new K at time_step for batch 1 {}: {}'.format(t, new_K[1, :, :, 0]))

    # ok, test passed.
