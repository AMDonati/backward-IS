import torch
import torch.nn.functional as F

def resample(params, I):
    '''
    :params: shape (B,P,H)
    :I: shape (B,P)
    '''
    I = I.unsqueeze(-1).repeat(1,1,params.size(-1))
    resampled_params = torch.gather(input=params, index=I, dim=1)
    return resampled_params

def estimation_function_X0(k, X):
    if k == 0:
        out = X
    else:
        out = 0.
    return out

def inv_tanh(x):
    return 1/2*torch.log((1+x)/(1-x))

def derive_tanh(x):
    return 1 - torch.pow(F.tanh(x), 2)

if __name__ == "__main__":

    # ---------- test of resample function -----------------------------------------------------------------------------------------------------------
    B = 2
    P = 4
    D = 2

    ind_matrix = torch.tensor([[1,1,2,3],[0,1,3,2]])

    K = torch.tensor([[[1,5],[2,6],[3,7],[4,8]],[[13,17], [14,18], [15,19], [16,20]]])

    truth = torch.tensor([[[2,6], [2,6], [3,7], [4,8]], [[13,17], [14,18], [16,20], [15,19]]])

    resampled_K = resample(K, ind_matrix)

    print("TRUTH:{}", truth)
    print("resampled_K:{}", resampled_K)

    # ok, test passed.

    # ------------- test of inv tanh x ----------------------------------------------------------------------
    X = torch.ones(size=(8,10,32))
    XX = F.tanh(inv_tanh(X))
    print(XX)