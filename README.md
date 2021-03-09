# backward-IS

### Code Structure
```
├── output       
|
├── data             
|
└── src   
 └── models  # package for stochastic RNNs (LSTM not fully implemented.)
    └── stochastic_RNN.py: # stochastic Recurrent Neural Network to generate (states, observations). Main functions:
      * forward(input, hidden): forward pass of the neural network
      * update_sigmas(sigma_h, sigma_y, sigma_init): # update the value of the three diagonal covariance matrix with non-zeros values after training. 
      * generate_observations(initial_input, seq_len, sigma_init, sigma_h, sigma_y, num_samples=100): # generate observations from initial observation $Y_0$
      * estimate_transition_density(particle, ancestor): compute q_k(\xi_{k}^i, \xi_{k-1}^J_k(j))
    
 └── smc: # package for SMC algorithms
      └── BootstrapFilter.py # class implementing a bootstrap filter. Main attributes:
      * RNN: stochastic RNN with learned parameter $\theta$ as the generative model 
      * num_particles: number of particles for the bootstrap filter algo. 
      Main functions: 
        * compute_filtering_weights(hidden, observations): compute the sampling weights for the bootstrap filter by computing a gaussian density function of mean (rnn.output_layer(hidden=\xi_{k}^l)) applied on observations = $Y_{k+1}$
        * get_new_particle(observation, next_observation, hidden, weights): compute $\xi_{k+1}^l$ from observation $Y_{k+1}$, hidden $\xi_k^l=h_k$, and previous weights $w_k$, and compute new weights $w_{k+1}$ from current next_observation $Y_{k+1}$ and $\xi_{k+1}^l$
      └── BackwardIS_smoother.py # class implementing the Backward Importance Sampling Smoother. Main Attributes:
      * Bootstrap filter 
      * Observations: Sequence of Observations $(Y_k)_{k=0}^n$
      * States: Sequence of States $(X_k)_{k=0}^n$ (only X_0 used). 
      * num_particles: number of particles for the bootstrap filter. 
      * backward_samples: number of samples for the backward simulation. 
     Main functions: 
          * init_particles: initialize ($\xi_0^l$, $w_0^l$, $\tau_0^l$)
          * update_tau(ancestors, particle, backward_indices, IS_weights, k): compute $\tau_k^l from $\tau_{k-1}^l, $w_{k-1]^l, $\xi_{k-1}^{J_k(j)}$ and from J_k(j), \Tilde(w)(l,j) for all j in 0,...,backward samples
          * estimate_conditional_expectation_of_function: compute $mathbb[E][X_0|Y_{0:n}]$ from $X_0$ and the sequence of observations $Y_{0:n}$
          * compute_mse_phi_X0(phi): $(mathbb[E][X_0|Y_{0:n}] - X_0)^2 with phi=mathbb[E][X_0|Y_{0:n}]$ 
       └── utils.py: 
            * function resample to resample trajectories
            * Inverse tanh function, and tanh derivative function to compute the transition density function of the stochastic RNN
            * function to estimate: $mathbb[E][X_0|Y_{0:n}]$
 └── train: # script to train the deterministic RNN 
 


# source files
```
