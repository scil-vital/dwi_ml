# -*- coding: utf-8 -*-

import numpy as np
import torch

"""
Variables:
    v = the normalized target
    mu = the mean
    kappa = the concentration parameter
    d = the dimension

Formulas:

    1. Probability function:
            P(v | mu, kappa) = C exp(kappa*mu^T*v)

        Where C = the distribution normalizing constant and I_n = the modified
        Bessel function at order n (see [3]).
            C(kappa) = (kappa)^(d/2-1) / ((2π)^d/2 * I_(d/2-1)(kappa))

        In our case, d=3:
            C = kappa / (2*pi*(exp(kappa) - exp(-kappa)))

    2. log-likelihood:
            log(P(v)) = log(C exp(kappa*mu^T*v))
                      = log(C) + kappa*mu^T*v

            log(C) = log(kappa) - log(2pi) - log(exp(kappa)-exp(-kappa)

    Refs:
    [1]: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    [2]: http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    [3]: https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1
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
