import torch
import torch.nn.functional as F
import math


def resample(params, I):
    '''
    :params: shape (B,P,H)
    :I: shape (B,P)
    '''
    I = I.unsqueeze(-1).repeat(1, 1, params.size(-1))
    resampled_params = torch.gather(input=params, index=I, dim=1)
    return resampled_params

def resample_1D(params, I):
    '''
    :params: shape (P,1)
    :I: shape (P)
    '''
    I = I.unsqueeze(1)
    resampled_params = torch.gather(input=params, index=I, dim=0)
    return resampled_params


def resample_all_seq(params, i_t):
    '''
    :param params > shape (P,J,S,1)
    :param i_t: current indice > shape (P,J)
    '''
    I = i_t.view(i_t.size(0), i_t.size(1), 1, 1).repeat(1, 1, params.size(-2), params.size(-1))
    resampled_params = torch.gather(input=params, index=I, dim=1)
    return resampled_params

def resample_all_seq_1D(params, i_t):
    '''
    :param params > shape (P,S,1)
    :param i_t: current indice > shape (P,1)
    '''
    I = i_t.unsqueeze(1).repeat(1, params.size(1),1) # shape (P,S,1)
    resampled_params = torch.gather(input=params, index=I, dim=0)
    return resampled_params


def estimation_function_X(k, X, index):
    if k == index:
        out = X
    else:
        out = X.new_zeros(X.size(0), X.size(1), X.size(2))
    return out


def inv_tanh(x):
    return 1 / 2 * torch.log((1 + x) / (1 - x))


def derive_tanh(x):
    return 1 - torch.pow(F.tanh(x), 2)


def log_gaussian_density_function(X, mean, covariance):
    ''' Compute the Gaussian Density Function with mean, diagonal covariance diag(covariance) at input X.
    :param X: tensor of shape (B, P, hidden_size)
    :param mean: tensor of shape (B, P, hidden_size)
    :param covariance: scalar value.
    '''
    distrib = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean,
                                                                         covariance_matrix=covariance * torch.eye(
                                                                             mean.size(-1)))
    log_d = distrib.log_prob(X)
    return log_d

def manual_log_density_function(X, mean, covariance):
    if len(covariance.size())>=2:
        part_1 = torch.log(2*math.pi*torch.linalg.det(covariance))
    else:
        part_1 = torch.log(2*math.pi*covariance)
    part_2 = 1/covariance.squeeze() *(X.squeeze()-mean.squeeze())**2
    log_distrib = -0.5 * (part_1 + part_2)
    return log_distrib


if __name__ == "__main__":

    # ---------- test of resample function -----------------------------------------------------------------------------------------------------------
    B = 2
    P = 4
    D = 2

    ind_matrix = torch.tensor([[1, 1, 2, 3], [0, 1, 3, 2]])

    K = torch.tensor([[[1, 5], [2, 6], [3, 7], [4, 8]], [[13, 17], [14, 18], [15, 19], [16, 20]]])

    truth = torch.tensor([[[2, 6], [2, 6], [3, 7], [4, 8]], [[13, 17], [14, 18], [16, 20], [15, 19]]])

    resampled_K = resample(K, ind_matrix)

    print("TRUTH:{}", truth)
    print("resampled_K:{}", resampled_K)

    # ok, test passed.

    # ----------------------- test of resample all past trajectories function ---------------------------------------------------------

    B = 2
    S = 3
    P = 4
    D = 1

    ind_matrix = torch.tensor([[[1, 1, 2, 2], [0, 0, 0, 0], [1, 1, 1, 0]],
                               [[0, 1, 3, 2], [3, 3, 2, 0], [1, 2, 3, 1]]])
    ind_matrix = ind_matrix.permute(0, 2, 1)
    # ind_matrix = tf.tile(tf.expand_dims(ind_matrix, axis=0), multiples=[B, 1, 1])  # (B,P,S)

    print('indices_matrices', ind_matrix[0, :, :].numpy())

    K = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                      [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    K = K.permute(0, 2, 1)
    K = K.unsqueeze(-1)  # (B,P,S,D=1)
    print('init K', K[0, :, :, 0])

    truth_t0_b1 = torch.tensor([[2], [2], [3], [3]])
    truth_t1_b1 = torch.tensor([[2, 5], [2, 5], [2, 5], [2, 5]])
    truth_t2_b1 = torch.tensor([[2, 5, 10], [2, 5, 10], [2, 5, 10], [2, 5, 9]])
    truth_t0_b2 = torch.tensor([[13], [14], [16], [15]])
    truth_t1_b2 = torch.tensor([[15, 20], [15, 20], [16, 19], [13, 17]])
    truth_t2_b2 = torch.tensor([[15, 20, 22], [16, 19, 23], [13, 17, 24], [15, 20, 22]])

    truth_t0 = torch.stack([truth_t0_b1, truth_t0_b2])
    truth_t1 = torch.stack([truth_t1_b1, truth_t1_b2])
    truth_t2 = torch.stack([truth_t2_b1, truth_t2_b2])

    new_K = K
    for t in range(S):
        i_t = ind_matrix[:, :, t]
        Kt = resample_all_seq(params=new_K[:, :, :t + 1, :], i_t=i_t)
        if t < (S - 1):
            new_K = torch.cat([Kt, K[:, :, t + 1, :].unsqueeze(dim=-2)], dim=-2)
        print('new K at time_step {} for batch 0 {}:'.format(t, Kt[0, :, :, 0]))
        print('new K at time_step {} for batch 1 {}:'.format(t, Kt[1, :, :, 0]))

    # ------------- test of inv tanh x ----------------------------------------------------------------------
    X = torch.ones(size=(8, 10, 32))
    XX = F.tanh(inv_tanh(X))
    print(XX)

    # void
    # updateTauEStep_IS(const
    # unsigned
    # int & childIndex,
    # const
    # unsigned
    # int & childParticleIndex,
    # const
    # unsigned
    # int & ancestorParticleIndex,
    # const
    # double
    # IS_weight,
    # const
    # std::vector < SINE_POD > & testedModels){
    # for (unsigned int m = 0; m < testedModels.size(); m++){
    # SINE_POD model = testedModels[m];
    # double sampledLogQ = model.getModel().unbiasedLogDensityEstimate(particleSet(ancestorParticleIndex, childIndex - 1),
    # particleSet(childParticleIndex, childIndex),
    # observationTimes(childIndex - 1),
    # observationTimes(childIndex),
    # logDensitySampleSize,
    # skeletonSimulationMaxTry);
    # double logObsDensityTerm =  log(model.observationDensity(particleSet(childParticleIndex, childIndex),
    # observations(childIndex)));
    # tauEStep[m](childParticleIndex, childIndex) += IS_weight * (tauEStep[m](ancestorParticleIndex, childIndex - 1) +
    # sampledLogQ + logObsDensityTerm);
    # }
    # };

    # Rcpp::NumericVector
    # evalEStep_IS(const
    # std::vector < SINE_POD > & testedModels){
    #     unsigned
    # int
    # newParamSize = testedModels.size();
    # Rcpp::NumericVector
    # output(newParamSize);
    # initializeTauEStep(newParamSize); // Initialize
    # matrix
    # of
    # 0
    # setInitalParticles();
    # for (int k = 0; k < (observationSize - 1);k++){
    # propagateParticles(k);
    # // initializeBackwardSampling(k); // Samples of ancestor index is made here
    # Rcpp::
    #     NumericVector
    # currentWeights = particleFilteringWeights(Rcpp::_, k);
    # for (unsigned int i = 0; i < particleSize; i++){ // i indexes particles
    # // setDensityUpperBound(k + 1, i); // Density upperbound for particle xi_{k+1} ^ i
    # sum_IS_weights = 0;
    # double curParticle = particleSet(i, k + 1);
    # // Choosing ancestoir
    # Rcpp::
    #     IntegerVector
    # ancestInd = GenericFunctions::sampleReplace(particleIndexes,
    #                                             backwardSampleSize,
    #                                             currentWeights);
    # Rcpp::NumericVector
    # ancestPart(backwardSampleSize);
    # Rcpp::NumericVector
    # IS_weights(backwardSampleSize);
    # for (unsigned int l = 0; l < backwardSampleSize; l++){
    #     ancestPart(l) = particleSet(ancestInd(l), k);
    # IS_weights(l) = propModel.evalTransitionDensityUnit(ancestPart(l),
    # curParticle,
    # observationTimes(k),
    # observationTimes(k + 1),
    # densitySampleSize,
    # false);
    # sum_IS_weights += IS_weights(l);
    # }
    # IS_weights = IS_weights / sum_IS_weights;
    # for (unsigned int l = 0; l < backwardSampleSize; l++){
    # updateTauEStep_IS(k + 1, i, ancestInd(l), IS_weights(l), testedModels);
    # // k + 1 is the time index from which the backward is done,
    # // i is the corresponding particle of this generation
    # }
    # // std::
    #     cout << "sum of IS_Weights" << sum_IS_weights << std::endl;
    # }
    # }
    # // for (unsigned int m = 0; m < newParamSize; m++){
    #                                                   // std: :
    #     cout << "m = " << m << std::endl;
    # // Rcpp::NumericMatrix
    # taus = tauEStep[m];
    # // DebugMethods::debugprint(taus, "taus_MC");
    # //}
    # Rcpp::NumericVector
    # lastWeights = particleFilteringWeights(Rcpp::_, observationSize - 1);
    # for (int m = 0; m < newParamSize; m++){
    #     output[m] = sum(lastWeights * tauEStep[m](Rcpp::
    #     _, observationSize - 1));
    # }
    # return output;
    # }; // end
    # of
    # evalEstep
    # method;

    # Rcpp::NumericMatrix
    # GEM(const
    # Rcpp::NumericVector & observations,
    #       const
    # Rcpp::NumericVector & observationTimes,
    #       const
    # double
    # thetaStart, const
    # double
    # sigma2Start,
    # const
    # unsigned
    # int
    # nIterations = 20,
    #               const
    # unsigned
    # int
    # nModels = 5,
    #           const
    # double
    # randomWalkParam = 2,
    #                   const
    # unsigned
    # int
    # particleSize = 100,
    #                const
    # unsigned
    # int
    # densitySampleSize = 30,
    #                     const
    # unsigned
    # int
    # logDensitySampleSize = 30,
    #                        const
    # unsigned
    # int
    # backwardSampleSize = 2,
    #                      const
    # unsigned
    # int
    # backwardSamplingMaxTry = 100000000,
    #                          const
    # unsigned
    # int
    # skeletonSimulationMaxTry = 10000000,
    #                            const
    # bool
    # estimateTheta = true, const
    # bool
    # estimateSigma2 = true){
    #     Rcpp:: NumericMatrix
    # output(nIterations + 1, 2);
    # output.fill(sigma2Start);
    # output(0, 0) = thetaStart;
    # for (unsigned int iter = 1; iter < nIterations + 1; iter++){
    #     SINE_POD startModel(output(iter - 1, 0), output(iter - 1, 1));
    # std::
    #     vector < SINE_POD > testedModels(nModels + 1);
    # testedModels[0] = startModel;
    # for (int i = 1; i < (nModels + 1); i++){
    # SINE_POD tmp = generateModel(output(iter - 1, 0), output(iter - 1, 1), 1.0 / iter);
    # testedModels[i] = tmp;
    # }
    # ProposalSINEModel propModel(randomWalkParam, startModel, estimateTheta, estimateSigma2);
    # SDEParticleSmoother mySmoother(observations, observationTimes, propModel,
    # particleSize, densitySampleSize, logDensitySampleSize,
    # backwardSampleSize, backwardSamplingMaxTry, skeletonSimulationMaxTry);
    # Rcpp::
    #     NumericVector
    # E_value = mySmoother.evalEStep(testedModels);
    # unsigned
    # int
    # bestModelInd = Rcpp::which_max(E_value);
    # output(iter, 0) = testedModels[bestModelInd].getParams()[0];
    # output(iter, 1) = testedModels[bestModelInd].getParams()[1];
    # }
    # return output;
    # }
