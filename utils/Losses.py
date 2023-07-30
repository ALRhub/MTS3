import torch
import numpy as np
from matplotlib import pyplot as plt

def rmse(target, predicted):
    """Mean Squared Error"""
    return torch.sqrt(torch.mean((target - predicted)**2))

def mse(target, predicted):
    """Mean Squared Error"""
    #target = target[:,:1]
    return torch.mean((target - predicted)**2)

def gaussian_nll(target, predicted_mean, predicted_var):
    """Gaussian Negative Log Likelihood (assuming diagonal covariance)"""
    #target = target[:,:1]
    predicted_var += 1e-12
    mahal = (target - predicted_mean)**2 / predicted_var
    element_wise_nll = 0.5 * (torch.log(predicted_var) + np.log(2 * np.pi) + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-1)
    return torch.mean(sample_wise_error)

def root_mean_squared(pred, target, data=[], tar='observations', fromStep=0, denorma=False, plot=None):
    """
    root mean squared error
    :param target: ground truth positions
    :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
    :return: root mean squared error between targets and predicted mean, predicted variance is ignored
    """
    target = target[:,:1]
    pred = pred[..., :target.shape[-1]]

    sumSquare = 0
    count = 0
    if plot != None:
        for idx in range(target.shape[2]):
            plt.plot(target[3,:,idx],label='target')
            plt.plot(pred[3,:,idx],label='prediction')
            plt.legend()
            plt.show()

    # if denorma==True:
    #     pred = denorm(pred, data, tar)
    #     target = denorm(target, data, tar)



    #target = target[:, fromStep:, :]
   # pred = pred[:, fromStep:, :]
    numSamples = 1
    for dim in target.shape:
        numSamples = numSamples * dim
    #print('RMSE Samplesss......................................',numSamples)
    sumSquare = np.sum(np.sum(np.sum((target - pred) ** 2)))
    return np.sqrt(sumSquare / numSamples)