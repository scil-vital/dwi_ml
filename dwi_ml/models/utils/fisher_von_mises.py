# -*- coding: utf-8 -*-

import numpy as np
import torch

"""
The complete formulas and explanations are available in our doc:
https://dwi-ml.readthedocs.io/en/latest/formulas.html
"""


def fisher_von_mises_log_prob_vector(mus, kappa, targets):
    log_c = np.log(kappa) - np.log(2 * np.pi) - np.log(np.exp(kappa) -
                                                       np.exp(-kappa))
    log_prob = log_c + (kappa * (mus * targets).sum(axis=-1))
    return log_prob


def fisher_von_mises_log_prob(mus, kappa, targets, eps=1e-6):
    log_2pi = np.log(2 * np.pi).astype(np.float32)

    # Add an epsilon in case kappa is too small (i.e. a uniform
    # distribution)
    log_diff_exp_kappa = torch.log(torch.exp(kappa) - torch.exp(-kappa) + eps)
    log_c = torch.log(kappa) - log_2pi - log_diff_exp_kappa

    batch_dot_product = torch.sum(mus * targets, dim=1)

    log_prob = log_c + (kappa * batch_dot_product)

    return log_prob
