# -*- coding: utf-8 -*-

import numpy as np

"""
The complete formulas and explanations are available in our doc:
https://dwi-ml.readthedocs.io/en/latest/formulas.html
"""
d = 3


def independent_gaussian_log_prob(targets, mus, sigmas):
    """
    This function computes the log likelihood for **individual** multivariate
    Gaussians, computed from tensors.

    If K>1 (the number of Gaussians), then the variables mu and sigma contain
    the information for many Gaussians, and log likelihoods for each Gaussian
    are returned.

    Parameters
    ----------
    targets: torch.tensor
        The variable, of shape [batch_size, none, d], d=3.
    mus: torch.tensor
        The mean, of size [batch_size, n, d], d=3.
    sigmas: torch.tensor
        The standard deviation
    """
    log_2pi = np.log(2 * np.pi).astype(np.float32)
    squared_m = ((targets - mus) / sigmas).pow(2).sum(dim=-1)
    log_det = sigmas.log().sum(dim=-1)

    gaussians_log_prob = -0.5 * (d * log_2pi + squared_m) - log_det

    return gaussians_log_prob
