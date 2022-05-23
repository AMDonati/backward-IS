import numpy as np

alpha = 0.91
sigma = 1.0
beta = 0.5

X0 = np.random.normal(scale=sigma/np.sqrt(1-sigma**2))

seq_len = 10

X = X0

observations = np.zeros(seq_len)
for k in range(seq_len):
    next_X = alpha * X + np.random.normal(scale=sigma)
    Y = beta * np.exp(next_X/2)*np.random.normal()
    observations[k] = Y

